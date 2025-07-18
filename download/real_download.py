import cdsapi
import zipfile



def download(year, month, day):
    client = cdsapi.Client() # Making a connection to CDS, to fetch data.
    singlelevelfields_instant = [
                            '2m_temperature',
                        ]
    dataset_single = "reanalysis-era5-single-levels"
    request_single_instant = {
        "product_type": ["reanalysis"],
        "variable": singlelevelfields_instant,
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{da:02d}" for da in range(day[0], day[1] + 1)],
        "time": [f"{tm:02d}:00" for tm in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client.retrieve(dataset_single, request_single_instant, 'download/real-temp.nc')

if __name__ == "__main__":
    download(2024, 7, (9, 18))