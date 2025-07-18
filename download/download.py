import cdsapi
import zipfile



def download(year, month, day):
    client = cdsapi.Client() # Making a connection to CDS, to fetch data.
    singlelevelfields_instant = [
                            '10m_u_component_of_wind',
                            '10m_v_component_of_wind',
                            '2m_temperature',
                            'geopotential',
                            'land_sea_mask',
                            'mean_sea_level_pressure',
                            
                        ]
    dataset_single = "reanalysis-era5-single-levels"
    request_single_instant = {
        "product_type": ["reanalysis"],
        "variable": singlelevelfields_instant,
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}"],
        "time": ["12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }


    # The fields to be fetched from the pressure-level source.
    pressurelevelfields = [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"
    ]

    dataset_pres = "reanalysis-era5-pressure-levels"
    request_pres = {
        "product_type": ["reanalysis"],
        "variable": pressurelevelfields,
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}"],
        "time": ["12:00", "18:00"],
        "pressure_level": ["50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client.retrieve(dataset_single, request_single_instant, 'download/single-level.nc')
    client.retrieve(dataset_pres, request_pres, 'download/pressure-level.nc')

if __name__ == "__main__":
    download(2025, 7, 11)