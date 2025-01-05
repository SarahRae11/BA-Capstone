!pip install spiceypy
!pip install tqdm
!pip install spacepy
!pip install dtw
!pip install statsmodels

import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# read file
import zipfile
with zipfile.ZipFile('MMESH-main.zip', 'r') as zip:
    zip.printdir()
    zip.extractall()

text_file = zip.namelist()

import sys

for i in range(len(text_file)):
  sys.path.insert(0,str(zip.namelist()[i]))

# initiate configuration file
config_filepath = 'minimal_example_toml.toml'

import toml

with open('minimal_example_toml.toml', 'r') as f:
    config = toml.load(f)
    print(config)

reference_frame = 'SUN_INERTIAL'
observer = 'SUN'

import MMESH102 as mmesh
import data_and_model_functions as func
import matplotlib.pyplot as plt
#import plot_TaylorDiagram6 as TD

config_filepath = 'minimal_example_toml.toml'

mtraj = mmesh.MultiTrajectory(config_filepath)

def minimal_example():

    config_filepath = 'minimal_example_toml.toml'

    mtraj = mmesh.MultiTrajectory(config_filepath)

    traj = mmesh.Trajectory(config_filepath)

    for trajectory_name, trajectory in mtraj.trajectories.items():

        starttime = trajectory.start
        stoptime = trajectory.stop

        spacecraft = func.data(trajectory.source, starttime, stoptime)

        #Define variables 

        density, vel_mag_data, field_mag_data, dynamic_pressure = spacecraft

        #Create a dataframe

        dataframe = pd.concat([density, vel_mag_data, dynamic_pressure, field_mag_data], axis = 1)

        dataframe.columns =  ['n_tot', 'u_mag', 'p_dyn', 'B_mag']

        trajectory.data_index = dataframe.index 

        trajectory.addData(trajectory_name, dataframe)

        #Add padding to the model starttime and stoptime

        padding = dt.timedelta(days=1)

        for model_name, model_source in mtraj.model_sources.items():

            model = func.model(model_source, starttime - padding, stoptime + padding)

            density, vel_mag, field_mag, dynamic_p = model

            dataframe_model = pd.concat([density, vel_mag, dynamic_p, field_mag], axis = 1)

            dataframe_model.columns = ['n_tot', 'u_mag', 'p_dyn', 'B_mag']

            trajectory.addModel(model_name, dataframe_model, model_source=model_source)

        print('primary df', trajectory._primary_df)
        
        #Add context

        import context_functions as mmesh_c

        starttime = trajectory.start
        stoptime = trajectory.stop

        context_path = mtraj.filepaths['output']/'context/'
        context_fullfilepath = mmesh_c.PSP(mercury_context, PSP_context, starttime - padding, stoptime + padding, filepath = context_path)
        context_df = pd.read_csv(context_fullfilepath, parse_dates = True)
        context_df.set_index('datetime', inplace=True)
        context_df.index = pd.to_datetime(context_df.index)
        context_df = context_df.dropna()
        #context_df.index.name = None
        trajectory.context = context_df

        #Smooth the data then binarize the data and models

        smoothing_widths = trajectory.optimize_ForBinarization('u_mag', threshold = 1.5)

        trajectory.binarize('u_mag', smooth = smoothing_widths, sigma = 1)

            # connect shifts to padding/resolution in .toml
        dtw_stats = trajectory.find_WarpStatistics('jumps', ('u_mag', 'p_dyn'),
                                                    shifts=np.arange(-96, 96+6, 6), intermediate_plots=False)
    

    #   Write an equation describing the optimization equation
    #   This can be played with
        def optimization_eqn(df):
            f = df[('u_mag','r')] + (1 - (0.5*df[('u_mag','width_68')])/96.)
            return (f - np.min(f))/(np.max(f)-np.min(f))
        
        #   Plug in the optimization equation
        trajectory.optimize_Warp(optimization_eqn)
        
        #   This plot shows the result of the DTW optimization
        trajectory.plot_OptimizedOffset('jumps', 'u_mag' , fullfilepath=mtraj.filepaths['output']/'figures/fig04_DTWIllustration')

    for cast_name, cast in mtraj.cast_intervals.items():
        
        #   Fill with models...
        for model_name, model_source in mtraj.model_sources.items():
            
            model = func.model(model_source, cast.start - padding, cast.stop + padding)
            
            density, vel_mag, field_mag, dynamic_p = model

            dataframe_model = pd.concat([density, vel_mag, dynamic_p, field_mag], axis = 1)

            dataframe_model.columns = ['n_tot', 'u_mag', 'p_dyn', 'B_mag']

            dataframe_model.head()
            
            cast.addModel(model_name, dataframe_model, model_source=model_source)

        #   Fill with context...

        starttime = cast.start
        stoptime = cast.stop

        context_fullfilepath = mmesh_c.PSP(mercury_context, PSP_context, starttime - padding, stoptime + padding, filepath = context_path)

        cast.context = pd.read_csv(context_fullfilepath, index_col='datetime', parse_dates=True)
        

        #   Add data for comparison purposes only
        #   This data will remain unchanged
        spacecraft = func.data(cast.target, starttime, stoptime)

        #Define variables 

        density, vel_mag_data, field_mag_data, dynamic_pressure = spacecraft

        #Create a dataframe

        dataframe = pd.concat([density, vel_mag_data, dynamic_pressure, field_mag_data], axis = 1)

        dataframe.columns =  ['n_tot', 'u_mag', 'p_dyn', 'B_mag']

        cast.addData('PSP', dataframe)

    #   Now for the actual casting! Based on a formula describing how the context should inform the uncertainty model:
    #   This translates to:
    #       empirical time delta = a * (target_sun_earth_lat) + b * (solar_radio_flux) + c * (u_mag)
    formula = "empirical_time_delta ~ target_sun_earth_lat + solar_radio_flux + u_mag + recurrence_index + p_dyn + B_mag + n_tot"

    test = mtraj.linear_regression(formula)

    #mtraj.trajectories['PSP'].plot_TaylorDiagram(tag_name = 'u_mag', fig = None, ax = None)

    #   with_error? !!!! explain
    cast_stats = mtraj.cast_Models(with_error=True, min_rel_scale = 0.01)
    #cast_stats.to_csv('{}-{}_cast.csv'.format(starttime.strftime('%Y%m%d%H%M%S'),stoptime.strftime('%Y%m%d%H%M%S')))
    #mtraj.ensemble()

    #original_TS = mtraj.taylor_statistics(cast=False, with_error=False)
    #final_TS = mtraj.taylor_statistics(cast=True, with_error=True, n_mc=10)

    mtraj.save_MultiTrajectory()

    return mtraj

