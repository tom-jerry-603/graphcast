import dataclasses
import datetime
import pandas as pd
import functools
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import numpy as np
import pvlib
import xarray

def calculate_toa_incident_solar_radiation(times, lat, lon):
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.to_datetime(times)
    toa_data = xarray.open_dataset('download/toa.nc')
    time_coord = toa_data['valid_time'].values

    time_coord_mdh = np.array([
        f"{pd.Timestamp(d).month:02d}-{pd.Timestamp(d).day:02d}-{pd.Timestamp(d).hour:02d}"
        for d in time_coord
    ])

    times_mdh = np.array([
        f"{pd.Timestamp(d).month:02d}-{pd.Timestamp(d).day:02d}-{pd.Timestamp(d).hour:02d}"
        for d in times
    ])

    # Compare by month-day-hour
    mask = np.isin(time_coord_mdh, times_mdh)

    # Select data where mask is True
    toa_data = toa_data.isel(valid_time=mask)
    toa_data = toa_data.sortby('latitude', ascending=True)
    # Use pvlib to compute TOA irradiance
    print(toa_data["tisr"])
    return xarray.DataArray(
        toa_data["tisr"],
        dims=['time', 'lat', 'lon'],
        coords={'time': times, 'lat': lat, 'lon': lon},
        name='toa_incident_solar_radiation',
        attrs={'units': 'W/m^2', 'description': 'Top of atmosphere incident solar radiation'}
    )

def compute_year_day_progress(time):
    # Convert to pandas DatetimeIndex (or Series)
    time = pd.to_datetime(time)

    day_of_year = time.dayofyear
    # is_leap_year is a pandas attribute, can be accessed directly for DatetimeIndex
    # but to get days_in_year as array, use np.where on is_leap_year boolean mask
    days_in_year = np.where(time.is_leap_year, 366, 365)

    year_frac = (day_of_year - 1) / days_in_year

    seconds_in_day = 24 * 3600
    time_of_day_seconds = time.hour * 3600 + time.minute * 60 + time.second

    day_frac = time_of_day_seconds / seconds_in_day

    year_progress_sin = np.sin(2 * np.pi * year_frac).astype('float32')
    year_progress_cos = np.cos(2 * np.pi * year_frac).astype('float32')
    day_progress_sin = np.sin(2 * np.pi * day_frac).astype('float32')
    day_progress_cos = np.cos(2 * np.pi * day_frac).astype('float32')

    return year_progress_sin, year_progress_cos, day_progress_sin, day_progress_cos

def get_input():
    with open(r'download/pressure-level.nc', 'rb') as f:
        pressure_level_ds = xarray.load_dataset(f).compute()
    with open(r'download/single-instant.nc', 'rb') as f:
        single_level_ds = xarray.load_dataset(f).compute()
    with open(r'download/single-accum.nc', 'rb') as f:
        single_level_accum_ds = xarray.load_dataset(f).compute()

    single_level_ds['tp'] = single_level_accum_ds['tp'].drop_vars('expver').coarsen(valid_time=6, boundary='trim').sum()

    pressure_level_rename_map = {
        'z': 'geopotential',
        'q': 'specific_humidity',
        't': 'temperature',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind',
        'w': 'vertical_velocity'
    }
    single_level_rename_map = {
        'u10': '10m_u_component_of_wind',
        'v10': '10m_v_component_of_wind',
        't2m': '2m_temperature',
        'z':   'geopotential_at_surface',
        'lsm': 'land_sea_mask',
        'msl': 'mean_sea_level_pressure',
        'tp': 'total_precipitation_6hr'
    }
    pressure_level_ds_renamed = pressure_level_ds.rename_vars(pressure_level_rename_map)
    single_level_ds_renamed = single_level_ds.rename_vars(single_level_rename_map)
    
    single_level_ds_renamed = single_level_ds_renamed.rename({
        'valid_time': 'time',
        'latitude': 'lat',
        'longitude': 'lon'
    })

    pressure_level_ds_renamed = pressure_level_ds_renamed.rename({
        'valid_time': 'time',
        'pressure_level': 'level',
        'latitude': 'lat',
        'longitude': 'lon'
    })

    # Add batch dimension (size 1) to both datasets
    single_level_ds_renamed = single_level_ds_renamed.expand_dims('batch')
    pressure_level_ds_renamed = pressure_level_ds_renamed.expand_dims('batch')

    # Optionally convert coordinates to float32 and int32 to match target
    single_level_ds_renamed = single_level_ds_renamed.assign_coords({
        'lat': single_level_ds_renamed.lat.astype('float32'),
        'lon': single_level_ds_renamed.lon.astype('float32'),
    })

    pressure_level_ds_renamed = pressure_level_ds_renamed.assign_coords({
        'lat': pressure_level_ds_renamed.lat.astype('float32'),
        'lon': pressure_level_ds_renamed.lon.astype('float32'),
        'level': pressure_level_ds_renamed.level.astype('int32'),
    })

    # If you want to merge them into one dataset (variables coexist)
    combined_ds = xarray.merge([single_level_ds_renamed, pressure_level_ds_renamed])
    combined_ds = combined_ds.drop_vars(['expver', 'number'])
    year_sin, year_cos, day_sin, day_cos = compute_year_day_progress(combined_ds.time.values)
    toa_irrad = calculate_toa_incident_solar_radiation(
        times=combined_ds.time.values.astype('datetime64[ns]'),
        lat=combined_ds.lat.values,
        lon=combined_ds.lon.values
    )
    toa_irrad = toa_irrad.expand_dims(dim={"batch": 1})
    combined_ds = combined_ds.assign({
        'toa_incident_solar_radiation': toa_irrad,
        'year_progress_sin': (('batch', 'time'), np.tile(year_sin, (combined_ds.batch.size, 1))),
        'year_progress_cos': (('batch', 'time'), np.tile(year_cos, (combined_ds.batch.size, 1))),
        'day_progress_sin': (('batch', 'time', 'lon'), np.tile(np.array(day_sin)[:, None], (1, combined_ds.sizes['lon']))[np.newaxis, :, :]),
        'day_progress_cos': (('batch', 'time', 'lon'), np.tile(np.array(day_sin)[:, None], (1, combined_ds.sizes['lon']))[np.newaxis, :, :]),
    })
    combined_ds['geopotential_at_surface'] = combined_ds['geopotential_at_surface'].isel(batch=0, time=0)
    combined_ds['land_sea_mask'] = combined_ds['land_sea_mask'].isel(batch=0, time=0)
    combined_ds = combined_ds.sortby('lat', ascending=True)
    combined_ds = combined_ds.sortby('level', ascending=True)
    return combined_ds
    
# <xarray.Dataset> Size: 2GB
# Dimensions:                       (batch: 1, time: 2, lat: 721, lon: 1440, level: 37)
# Coordinates:
#   * lon                           (lon) float32 6kB 0.0 0.25 0.5 ... 359.5 359.8
#   * lat                           (lat) float32 3kB -90.0 -89.75 ... 89.75 90.0
#   * level                         (level) int32 148B 1 2 3 5 ... 950 975 1000
#   * time                          (time) timedelta64[ns] 16B -1 days +18:00:0...
# Dimensions without coordinates: batch
# Data variables: (12/18)
    # 2m_temperature                (batch, time, lat, lon) float32 8MB 250.7 ....
    # mean_sea_level_pressure       (batch, time, lat, lon) float32 8MB 9.931e+...
    # 10m_v_component_of_wind       (batch, time, lat, lon) float32 8MB -0.4393...
    # 10m_u_component_of_wind       (batch, time, lat, lon) float32 8MB 1.309 ....
    # total_precipitation_6hr       (batch, time, lat, lon) float32 8MB 0.00043...
    # temperature                   (batch, time, level, lat, lon) float32 307MB ...
    # geopotential                  (batch, time, level, lat, lon) float32 307MB ...
    # u_component_of_wind           (batch, time, level, lat, lon) float32 307MB ...
    # v_component_of_wind           (batch, time, level, lat, lon) float32 307MB ...
    # vertical_velocity             (batch, time, level, lat, lon) float32 307MB ...
    # specific_humidity             (batch, time, level, lat, lon) float32 307MB ...
    # toa_incident_solar_radiation  (batch, time, lat, lon) float32 8MB 1.981e+...
    # year_progress_sin             (batch, time) float32 8B 0.006986 0.01129
    # year_progress_cos             (batch, time) float32 8B 1.0 0.9999
    # day_progress_sin              (batch, time, lon) float32 12kB 0.0 ... 1.0
    # day_progress_cos              (batch, time, lon) float32 12kB 1.0 ... 0.0...
    # geopotential_at_surface       (lat, lon) float32 4MB 2.735e+04 ... -0.07617
    # land_sea_mask                 (lat, lon) float32 4MB 1.0 1.0 1.0 ... 0.0 0.0


def get_targets(timesteps = 48):
    lat = np.linspace(-90, 90, 721, dtype=np.float32)
    lon = np.linspace(0, 359.75, 1440, dtype=np.float32)
    level = np.array([1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,  125,  
                    150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600,  
                    650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975,  
                    1000], dtype=np.int32)
    time = np.array([np.timedelta64((i+1) * 6, 'h') for i in range(timesteps)])  # 0h, 6h, 12h, 18h

    # Dimensions
    dims_2d = ("batch", "time", "lat", "lon")
    dims_3d = ("batch", "time", "level", "lat", "lon")

    shape_2d = (1, timesteps, 721, 1440)
    shape_3d = (1, timesteps, 37, 721, 1440)

    # Dummy data arrays (float32)
    targets = xarray.Dataset(
        data_vars={
            "2m_temperature": (dims_2d, np.random.rand(*shape_2d).astype(np.float32)),
            "mean_sea_level_pressure": (dims_2d, np.random.rand(*shape_2d).astype(np.float32)),
            "10m_u_component_of_wind": (dims_2d, np.random.rand(*shape_2d).astype(np.float32)),
            "10m_v_component_of_wind": (dims_2d, np.random.rand(*shape_2d).astype(np.float32)),
            "total_precipitation_6hr": (dims_2d, np.random.rand(*shape_2d).astype(np.float32)),
            "temperature": (dims_3d, np.random.rand(*shape_3d).astype(np.float32)),
            "geopotential": (dims_3d, np.random.rand(*shape_3d).astype(np.float32)),
            "u_component_of_wind": (dims_3d, np.random.rand(*shape_3d).astype(np.float32)),
            "v_component_of_wind": (dims_3d, np.random.rand(*shape_3d).astype(np.float32)),
            "vertical_velocity": (dims_3d, np.random.rand(*shape_3d).astype(np.float32)),
            "specific_humidity": (dims_3d, np.random.rand(*shape_3d).astype(np.float32)),
        },
        coords={
            "lat": ("lat", lat),
            "lon": ("lon", lon),
            "level": ("level", level),
            "time": ("time", time),
        }
    )
    return targets
# <xarray.Dataset> Size: 943MB
# Dimensions:                  (batch: 1, time: 1, lat: 721, lon: 1440, level: 37)
# Coordinates:
#   * lon                      (lon) float32 6kB 0.0 0.25 0.5 ... 359.5 359.8
#   * lat                      (lat) float32 3kB -90.0 -89.75 -89.5 ... 89.75 90.0
#   * level                    (level) int32 148B 1 2 3 5 7 ... 925 950 975 1000
#   * time                     (time) timedelta64[ns] 8B 06:00:00
# Dimensions without coordinates: batch
# Data variables:
#     2m_temperature           (batch, time, lat, lon) float32 4MB 248.3 ... 247.4
#     mean_sea_level_pressure  (batch, time, lat, lon) float32 4MB 9.983e+04 .....
#     10m_v_component_of_wind  (batch, time, lat, lon) float32 4MB -0.04184 ......
#     10m_u_component_of_wind  (batch, time, lat, lon) float32 4MB 0.04857 ... ...
#     total_precipitation_6hr  (batch, time, lat, lon) float32 4MB 7.443e-05 .....
#     temperature              (batch, time, level, lat, lon) float32 154MB 287...
#     geopotential             (batch, time, level, lat, lon) float32 154MB 4.9...
#     u_component_of_wind      (batch, time, level, lat, lon) float32 154MB -0....
#     v_component_of_wind      (batch, time, level, lat, lon) float32 154MB 0.0...
#     vertical_velocity        (batch, time, level, lat, lon) float32 154MB -0....
#     specific_humidity        (batch, time, level, lat, lon) float32 154MB 3.9...

def get_forcings(year, month, day, timesteps = 48):
    start_time = np.datetime64(f"{year}-{month:02d}-{day:02d}T00:00")
    time = np.array([np.timedelta64((i+1) * 6, 'h') for i in range(timesteps)])

    time_static = start_time + np.arange(timesteps) * np.timedelta64(6, 'h')
    # time = np.array([np.timedelta64((i+1) * 6, 'h') for i in range(4)])
    lat = np.linspace(-90, 90, 721).astype(np.float32)
    lon = np.linspace(0, 359.75, 1440).astype(np.float32)

    year_sin, year_cos, day_sin, day_cos = compute_year_day_progress(time_static)

    # Compute TOA incident solar radiation
    toa_irrad = calculate_toa_incident_solar_radiation(time_static, lat, lon)
    toa_irrad = np.expand_dims(toa_irrad.data, axis=0)
    # Build xarray Dataset
    forcings = xarray.Dataset(
        data_vars={
            'toa_incident_solar_radiation': (('batch', 'time', 'lat', 'lon'), toa_irrad),
            'year_progress_sin': (('batch', 'time'), np.tile(year_sin, (1, 1))),
            'year_progress_cos': (('batch', 'time'), np.tile(year_cos, (1, 1))),
            'day_progress_sin': (('batch', 'time', 'lon'), np.tile(np.array(day_sin)[:, None], (1, 1440))[np.newaxis, :, :]),
            'day_progress_cos': (('batch', 'time', 'lon'), np.tile(np.array(day_cos)[:, None], (1, 1440))[np.newaxis, :, :]),
        },
        coords={
            'time': time,
            'lat': lat,
            'lon': lon,
        }
    )
    return forcings

# <xarray.Dataset> Size: 4MB
# Dimensions:                       (batch: 1, time: 1, lat: 721, lon: 1440)
# Coordinates:
#   * lon                           (lon) float32 6kB 0.0 0.25 0.5 ... 359.5 359.8
#   * lat                           (lat) float32 3kB -90.0 -89.75 ... 89.75 90.0
#   * time                          (time) timedelta64[ns] 8B 06:00:00
# Dimensions without coordinates: batch
# Data variables:
#     toa_incident_solar_radiation  (batch, time, lat, lon) float32 4MB 1.978e+...
#     year_progress_sin             (batch, time) float32 4B 0.01559
#     year_progress_cos             (batch, time) float32 4B 0.9999
#     day_progress_sin              (batch, time, lon) float32 6kB -8.742e-08 ....
#     day_progress_cos              (batch, time, lon) float32 6kB -1.0 ... -1.0


