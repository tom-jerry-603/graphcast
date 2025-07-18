import cdsapi

client = cdsapi.Client()

def download(year, month, day):
    request_single_instant = {
        "product_type": ["reanalysis"],
        "variable": [
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        '2m_temperature',
                        'geopotential',
                        'land_sea_mask',
                        'mean_sea_level_pressure',
                    ],
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}"],
        "time": ["12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    request_single_accum = {
        "product_type": ["reanalysis"],
        "variable": "total_precipitation",
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}"],
        "time": [f"{tm:02d}:00" for tm in range(7, 19)],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    request_pres = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity"
        ],
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}"],
        "time": ["12:00", "18:00"],
        "pressure_level": [f"{i}" for i in [1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,  125,  
                                            150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600,  
                                            650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975,  
                                            1000]
                            ],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client.retrieve("reanalysis-era5-single-levels", request_single_instant, 'download/single-instant.nc')
    client.retrieve("reanalysis-era5-single-levels", request_single_accum, 'download/single-accum.nc')
    client.retrieve("reanalysis-era5-pressure-levels", request_pres, 'download/pressure-level.nc')

if __name__ == "__main__":
    download(2025, 7, 11)