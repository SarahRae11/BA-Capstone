import pandas as pd

def PSP(mercury_context, PSP_context, starttime, stoptime, filepath = ''):

    mercury_df = pd.read_csv(mercury_context)

    mercury_df = mercury_df.set_index(mercury_df['datetime'])

    mercury_df.index = pd.to_datetime(mercury_df.index)

    mercury_df = mercury_df.loc[starttime:stoptime]

    mercury_df = mercury_df.reset_index(drop = True)

    psp_df = pd.read_table(PSP_context, comment = '#', sep = '\s+')

    psp_df.columns = ['datetime', 'X', 'Y', 'Z', 'target_sun_earth_r', 'target_sun_earth_lon', 'target_sun_earth_lat']

    for i in range(len(psp_df['datetime'])):
        psp_df.loc[i, 'datetime'] = psp_df.loc[i, 'datetime'].replace('T', ' ')

    psp_df_1 = psp_df[['datetime', 'target_sun_earth_r', 'target_sun_earth_lon', 'target_sun_earth_lat']]

    psp_df_1 = psp_df_1.set_index(psp_df_1['datetime'])

    psp_df_1.index = pd.to_datetime(psp_df_1.index)

    psp_df_1 = psp_df_1.loc[starttime:stoptime]

    PSP_df = psp_df_1.reset_index(drop = True)

    PSP_df['solar_radio_flux'] = mercury_df['solar_radio_flux']
    PSP_df['smooth_solar_radio_flux'] = mercury_df['smooth_solar_radio_flux']
    PSP_df['recurrence_index'] = mercury_df['recurrence_index']

    PSP_df = PSP_df.set_index(PSP_df['datetime'])

    PSP_df.index = pd.to_datetime(PSP_df.index)

    PSP_df = PSP_df.drop('datetime', axis = 1)

    PSP_df_resampled = PSP_df.resample('180s').mean()

    PSP_df = PSP_df_resampled.interpolate(method='linear')

    filename = '{}-{}_Context.csv'.format(starttime.strftime('%Y%m%d%H%M%S'),stoptime.strftime('%Y%m%d%H%M%S'))

    fullfilepath = filepath / filename
    PSP_df.to_csv(str(fullfilepath), index_label='datetime')

    return fullfilepath

def mercury(mercury_context, starttime, stoptime):

    mercury_df = pd.read_csv(mercury_context)

    mercury_df = mercury_df.set_index(mercury_df['datetime'])

    mercury_df = mercury_df.loc[starttime:stoptime]

    mercury_df = mercury_df.reset_index(drop = True)

    return mercury_df