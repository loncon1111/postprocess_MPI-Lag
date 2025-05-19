import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'convective_available_potential_energy', 'instantaneous_moisture_flux', 'total_column_water',
            'total_column_water_vapour',
        ],
        'year': '2017',
        'month': '07',
        'day': [
            '15', '20', '22',
            '25', '28',
        ],
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
    },
    'download.nc')

