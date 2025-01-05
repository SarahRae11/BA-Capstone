!pip install sunpy
!pip install seaborn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# create dataframe of Mercury's orbit
import zipfile
with zipfile.ZipFile('mercury_orbit.zip', 'r') as zip1:
    zip1.extractall()

text_file = str(zip1.namelist()[0])

mercury_df = pd.read_table(text_file, comment = '#', sep = '\s+')
mercury_df.columns = ['Time', 'X', 'Y', 'Z']
mercury_df.head()

# read file and create dataframe of Parker Solar Probe's orbit
file = 'psp_hci_x_y_z_r_lon_lat_20180813-20240814.txt'
opened_file = open('psp_hci_x_y_z_r_lon_lat_20180813-20240814.txt', 'r')
content = opened_file.read()
print(content)

df = pd.read_table(file, comment = '#', sep = '\s+')
df.columns = ['Time', 'X', 'Y', 'Z', 'R', 'Longitude', 'Latitude']
df.head()

# Plot PSP's orbit
plt.plot(df['X'], df['Y'])
plt.xlabel('X (AU)')
plt.ylabel('Y (AU)')
sun = '\u2600'
plt.text(0 , 0 , sun , fontsize=40, ha='center', va='center', c = 'orange')
plt.grid()
plt.title('Plan of PSP Orbit in HCI Coords')
plt.show()
plt.plot(df['Y'], df['Z'])
plt.xlabel('Y (AU)')
plt.ylabel('Z (AU)')
plt.text(0 , 0 , sun , fontsize=40, ha='center', va='center', c = 'orange')
plt.grid()
plt.title('Side Elevation of PSP Orbit in HCI Coords')
plt.show()
plt.plot(df['X'], df['Z'])
plt.xlabel('X (AU)')
plt.ylabel('Z (AU)')
plt.text(0 , 0 , sun , fontsize=40, ha='center', va='center', c = 'orange')
plt.grid()
plt.title('Front Elevation of PSP Orbit in HCI Coords')

# To colour the plot
x = df['X']
y = df['Y']
z = df['Z']

def combined_color(x,z):
    x_norm = (x - x.min()) / (x.max() - x.min())
    z_norm = (z - z.min()) / (z.max() - z.min())
    return (x_norm + z_norm) / 2

# Plot PSP and Mercury orbit's together
fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(df['Y'], df['X'], df['Z'], c=combined_color(df['X'],df['Y']), cmap='hot', label='PSP Orbit', s = 2)
scat2 = ax.scatter(mercury_df['Y'], mercury_df['X'], mercury_df['Z'], c=combined_color(mercury_df['X'], mercury_df['Y']), cmap='ocean', label = 'Mercury Orbit', s = 2)
ax.text(0 , 0 , 0, sun , fontsize=40, ha='center', va='center', c = 'orange')
ax.set_xlabel('AU')
ax.set_ylabel('AU')
ax.set_zlabel('AU')
plt.title('3D Plot of PSP and Mercury Orbits in Heliocentric Inertial Coords')
plt.legend()
plt.show()

# Define where we want data from PSP's orbit
filtered_latitude = df[(df['Latitude'] >= -3.4) & (df['Latitude'] <= 3.4)]
norm = np.sqrt(x**2 + y**2 + z**2)
filtered_radius = filtered_latitude[(filtered_latitude['R'] >= 0.307) & (filtered_latitude['R'] <= 0.467)]

# Plot the final relevant date ranges 
fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_radius['Y'], filtered_radius['X'], filtered_radius['Z'], c=combined_color(filtered_radius['X'], filtered_radius['Y']), cmap='hot', s = 3)
ax.text(0 , 0 , 0, sun , fontsize=40, ha='center', va='center', c = 'orange')
ax.set_xlabel('AU')
ax.set_ylabel('AU')
ax.set_zlabel('AU')
plt.title("3D Plot of PSP's Orbit at ~Mercury's distance from Sun in HCI coords")

# Reset index 
filtered_radius = filtered_radius.reset_index(drop=True) 

# Get dates and times of relevant date ranges 
def split_time(df, gap_threshold):

    #Change time column to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    #Find difference in times
    time_diff = df['Time'].diff()

    #Find indices of time values that are greater than 2 hours in difference    
    gap_indices = time_diff[time_diff > pd.Timedelta(gap_threshold)].index

    print(gap_indices)

    #for x in gap_indices:
    #    print(time_diff[x-1])
    #    print(time_diff[x])
    #    print(time_diff[x+1])

    for x in gap_indices:
        print(df['Time'][x-1])
        print(df['Time'][x])
        print(df['Time'][x+1])

    #Split dataframe at gap indices
    splits = []
    start_idx = 0

    for idx in gap_indices:
        splits.append(df['Time'].iloc[start_idx:idx])
        start_idx = idx
    
    #Add the final segment
    splits.append(df['Time'].iloc[start_idx:])
    
    print('Len of splits:', len(splits))

    times = []

    for idx in gap_indices:
        times.append(df['Time'][idx])

    print(times)
    
    return splits

# Print
split_time(filtered_radius, '2H')[1]