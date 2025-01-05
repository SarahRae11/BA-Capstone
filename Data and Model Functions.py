def data(data, starttime, stoptime):

    #Install packages

    from spacepy import pycdf
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import datetime

    #Extract text file from zipfile

    import zipfile
    with zipfile.ZipFile(data, 'r') as zip:
        zip.extractall()

    text_file = str(zip.namelist()[0])

    #Create data frame

    df = pd.read_table(text_file, comment = '#', sep = '\s+')

    #Label columns

    df.columns = ['Date & Time', 'Density', 'Vel 1', 'Vel 2', 'Vel 3', 'Mag 1', 'Mag 2', 'Mag 3']

    #Change to datetime format

    pd.to_datetime(df['Date & Time'])

    ####Change time to DatetimeIndex of the data frame

    #df = df.set_index([pd.to_datetime(df['Date & Time']), df['Density']])

    #df = df.drop('Date & Time', axis=1)

    #df = df.drop('Density', axis=1)

    df = df.set_index(df['Date & Time'])

    df.index = pd.to_datetime(df.index)

    df = df.drop('Date & Time', axis=1)

    #Resample the data frame

    df = df.resample('180s').mean()

    #Define starttime and stoptime

    df = df.loc[starttime:stoptime]

    #Define velocity magnitude

    vel_mag_data = np.sqrt(df['Vel 1']**2 + df['Vel 2']**2 + df['Vel 3']**2)

    #Define magnetic field magnitude

    field_mag_data = np.sqrt(df['Mag 1']**2 + df['Mag 2']**2 + df['Mag 3']**2)

    #Define dynamic pressure

    proton_mass = 1.6726219e-27 #kg
    density_m_cubed = df['Density']*1e+6 #/m^3
    dynamic_pressure = 1e+9*proton_mass*density_m_cubed*(vel_mag_data*1000)**2

    #Return variables

    return df['Density'], vel_mag_data, field_mag_data, dynamic_pressure

    #Plot the data against time and as histograms
    '''
    plt.plot(df.index, df['Density'])
    plt.ylabel('Density (cm^-3)')
    plt.xlabel('Time')
    plt.xticks(rotation = 90)
    plt.show()
    plt.plot(df.index, vel_mag_data)
    plt.ylabel('Velocity Magnitude (km/s)')
    plt.xlabel('Time')
    plt.xticks(rotation = 90)
    plt.show()
    plt.plot(df.index, field_mag_data)
    plt.ylabel('Magnetic Field Magnitude (nT)')
    plt.xlabel('Time')
    plt.xticks(rotation = 90)
    plt.show()
    plt.plot(df.index, dynamic_pressure)
    plt.ylabel('Dynamic Pressure (nPa)')
    plt.xlabel('Time')
    plt.xticks(rotation = 90)
    plt.show()
    
    plt.hist(df['Density'], bins=100)
    plt.show()
    plt.hist(vel_mag_data, bins=100)
    plt.show()
    plt.hist(field_mag_data, bins=100)
    plt.show()
    plt.hist(dynamic_pressure, bins=100)
    plt.show()
    '''
    
def model(model, starttimemodel, stoptimemodel):

    #Install packages

    from spacepy import pycdf
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import datetime
    
    #Extract text file from zipfile

    import zipfile
    with zipfile.ZipFile(model, 'r') as zip:
        zip.extractall()

    text_file = str(zip.namelist()[0])

    #Create data frame

    model_df = pd.read_table(text_file, comment = '#', sep = '\s+')
    
    #Label columns

    model_df.columns = ['Date & Time', 'Density', 'Vel 1', 'Vel 2', 'Dynamic Pressure', 'Mag 1', 'Mag 2']

    #Change to datetime format

    pd.to_datetime(model_df['Date & Time'])

    #Change time to DatetimeIndex of the data frame

    model_df = model_df.set_index(model_df['Date & Time'])

    model_df.index = pd.to_datetime(model_df.index)

    model_df = model_df.drop('Date & Time', axis=1)

    #Interpolate the model

    model_df_resampled = model_df.resample('180s').mean()

    model_df = model_df_resampled.interpolate(method='linear')

    #Define starttime and stoptime

    model_df = model_df.loc[starttimemodel:stoptimemodel]

    #Define velocity magnitude

    vel_mag = np.sqrt(model_df['Vel 1']**2 + model_df['Vel 2']**2)

    #Define magnetic field magnitude

    field_mag = np.sqrt(model_df['Mag 1']**2 + model_df['Mag 2']**2)


    #Plot the model against time and as histograms
    '''
    plt.plot(model_df.index, model_df['Density'],label='Model')
    plt.ylabel('Density cm^(-3)')
    plt.xlabel('Time')
    plt.xticks(rotation = 90)
    plt.legend()
    plt.show()

    plt.plot(model_df.index, vel_mag, label='Model')
    plt.xticks(rotation = 90)
    plt.ylabel('Velocity Magnitude (km/s)')
    plt.xlabel('Time')
    plt.xticks(rotation = 90)
    plt.legend()
    plt.show()

    plt.plot(model_df.index, field_mag, label='Model')
    plt.xticks(rotation = 90)
    plt.ylabel('Magnetic Field Magnitude nT')
    plt.xlabel('Time')
    plt.show()

    plt.plot(model_df.index, model_df['Dynamic Pressure'], label='Model')
    plt.xticks(rotation = 90)
    plt.ylabel('Dynamic Pressure (nPa)')
    plt.xlabel('Time')
    plt.xticks(rotation = 90)
    plt.show()
    '''
    #return 

    return model_df['Density'], vel_mag, field_mag, model_df['Dynamic Pressure']
