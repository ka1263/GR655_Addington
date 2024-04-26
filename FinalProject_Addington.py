# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:06:32 2024

@author: kayle
"""

#SOUNDING via dropsonde (LFQ) ---------------------------------------------------------
#import necessary modules
import metpy
from datetime import datetime
import os
import scipy
import netCDF4
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import pandas_dataframe_to_unit_arrays, units
import numpy as np

#read in dropsonde data
dropsonde = netCDF4.Dataset('D20230911_092644_PQC.nc') 

#save dropsonde data
z = dropsonde.variables['gpsalt']
z_profile_mask = z[0:]
z_profile = []
for i in range(len(z_profile_mask)):
   if z_profile_mask.mask[i] == False:
        z_profile = np.append(z_profile, z_profile_mask[i])

launch_time = dropsonde.variables['time']
launch_time_mask = launch_time[0:]

i=0
launch_time_list = []
for i in range(len(launch_time_mask)):
    if z_profile_mask.mask.data[i] == False:
        launch_time_list = np.append(launch_time_list, launch_time_mask.data[i])

#create time variable
flight_start_time = launch_time.units

YYYY = flight_start_time[14:18]
DD = flight_start_time[22:24]
MM = flight_start_time[19:21]
HH = flight_start_time[25:27]
MIN = flight_start_time[28:30]
SS = flight_start_time[31:33]

flight_start_trakfile = MM + '/' + DD + '/' +YYYY
flight_start_trakfile_time = HH + ':' + MIN + ':' + SS

pressure = dropsonde.variables['pres']
pressure_profile_mask = pressure[0:]
pressure_profile = []
for i in range(len(pressure_profile_mask)):
   if pressure_profile_mask.mask[i] == False:
        pressure_profile = np.append(pressure_profile, pressure_profile_mask[i])
        
uwind = dropsonde.variables['u_wind']
uwind_mask = uwind[0:]
uwind_profile = []
for i in range(len(uwind_mask)):
   if uwind_mask.mask[i] == False:
        uwind_profile = np.append(uwind_profile, uwind_mask[i])

vwind = dropsonde.variables['v_wind']
vwind_profile_mask = vwind[0:]
vwind_profile = []
for i in range(len(vwind_profile_mask)):
   if vwind_profile_mask.mask[i] == False:
        vwind_profile = np.append(vwind_profile, vwind_profile_mask[i])
        
dp = dropsonde.variables['dp']
dp_profile_mask = dp[0:]
dp_profile = []
for i in range(len(dp_profile_mask)):
   if dp_profile_mask.mask[i] == False:
        dp_profile = np.append(dp_profile, dp_profile_mask[i])
        
t = dropsonde.variables['tdry']
t_profile_mask = t[0:]
t_profile = []
for i in range(len(t_profile_mask)):
   if t_profile_mask.mask[i] == False:
        t_profile = np.append(t_profile, t_profile_mask[i])
        
#ensure data are the same same size
p = pressure_profile[0:len(dp_profile)]
T = t_profile[0:len(dp_profile)]
Td = dp_profile[0:len(dp_profile)]
u = uwind_profile[0:len(dp_profile)]
v = vwind_profile[0:len(dp_profile)]

#plot figure
fig = plt.figure(figsize=(8, 8))

# Initiate the skew-T plot type from MetPy class loaded earlier
skew = SkewT(fig, rotation=45)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
#skew.plot_barbs(p[::25], u[::25], v[::25], y_clip_radius=0.03)

# Set some appropriate axes limits for x and y
skew.ax.set_xlim(-20, 40)
skew.ax.set_ylim(1020, 100)

# Add the relevant special lines to plot throughout the figure
skew.plot_dry_adiabats(t0=np.arange(233, 533, 10) * units.K,
                       alpha=0.25, color='orangered')
skew.plot_moist_adiabats(t0=np.arange(233, 400, 5) * units.K,
                         alpha=0.25, color='tab:green')

plt.title("LFQ: "+flight_start_trakfile)
plt.savefig('Final_Project_Addington_LFQSounding.png')
plt.show()
plt.close()


#SOUNDING 2 via dropsonde (RRQ) -------------------------------------------------------
#import necessary modules
from datetime import datetime
import os
import scipy
import metpy
import netCDF4
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import pandas_dataframe_to_unit_arrays, units
import numpy as np

#read in dropsonde data
dropsonde = netCDF4.Dataset('D20230911_080415_PQC.nc') 

#save dropsonde data
z = dropsonde.variables['gpsalt']
z_profile_mask = z[0:]
z_profile = []
for i in range(len(z_profile_mask)):
   if z_profile_mask.mask[i] == False:
        z_profile = np.append(z_profile, z_profile_mask[i])

launch_time = dropsonde.variables['time']
launch_time_mask = launch_time[0:]

i=0
launch_time_list = []
for i in range(len(launch_time_mask)):
    if z_profile_mask.mask.data[i] == False:
        launch_time_list = np.append(launch_time_list, launch_time_mask.data[i])

#create time variable
flight_start_time = launch_time.units

YYYY = flight_start_time[14:18]
DD = flight_start_time[22:24]
MM = flight_start_time[19:21]
HH = flight_start_time[25:27]
MIN = flight_start_time[28:30]
SS = flight_start_time[31:33]

flight_start_trakfile = MM + '/' + DD + '/' +YYYY
flight_start_trakfile_time = HH + ':' + MIN + ':' + SS

pressure = dropsonde.variables['pres']
pressure_profile_mask = pressure[0:]
pressure_profile = []
for i in range(len(pressure_profile_mask)):
   if pressure_profile_mask.mask[i] == False:
        pressure_profile = np.append(pressure_profile, pressure_profile_mask[i])
        
uwind = dropsonde.variables['u_wind']
uwind_mask = uwind[0:]
uwind_profile = []
for i in range(len(uwind_mask)):
   if uwind_mask.mask[i] == False:
        uwind_profile = np.append(uwind_profile, uwind_mask[i])

vwind = dropsonde.variables['v_wind']
vwind_profile_mask = vwind[0:]
vwind_profile = []
for i in range(len(vwind_profile_mask)):
   if vwind_profile_mask.mask[i] == False:
        vwind_profile = np.append(vwind_profile, vwind_profile_mask[i])
        
dp = dropsonde.variables['dp']
dp_profile_mask = dp[0:]
dp_profile = []
for i in range(len(dp_profile_mask)):
   if dp_profile_mask.mask[i] == False:
        dp_profile = np.append(dp_profile, dp_profile_mask[i])
        
t = dropsonde.variables['tdry']
t_profile_mask = t[0:]
t_profile = []
for i in range(len(t_profile_mask)):
   if t_profile_mask.mask[i] == False:
        t_profile = np.append(t_profile, t_profile_mask[i])

#ensure data are the same same size
p = pressure_profile[0:len(dp_profile)]
T = t_profile[0:len(dp_profile)]
Td = dp_profile[0:len(dp_profile)]
u = uwind_profile[0:len(dp_profile)]
v = vwind_profile[0:len(dp_profile)]

#plot figure
fig = plt.figure(figsize=(8, 8))

# Initiate the skew-T plot type from MetPy class loaded earlier
skew = SkewT(fig, rotation=45)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
#skew.plot_barbs(p[::25], u[::25], v[::25], y_clip_radius=0.03)

# Set some appropriate axes limits for x and y
skew.ax.set_xlim(-20, 40)
skew.ax.set_ylim(1020, 100)

# Add the relevant special lines to plot throughout the figure
skew.plot_dry_adiabats(t0=np.arange(233, 533, 10) * units.K,
                       alpha=0.25, color='orangered')
skew.plot_moist_adiabats(t0=np.arange(233, 400, 5) * units.K,
                         alpha=0.25, color='tab:green')

plt.title("RRQ: "+flight_start_trakfile)
plt.savefig('Final_Project_Addington_RRQSounding.png')
plt.show()
plt.close()


#SOUNDING 3 via dropsonde (RFQ) ---------------------------------------------------------
#import necessary modules
from datetime import datetime
import os
import scipy
import metpy
import netCDF4
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import pandas_dataframe_to_unit_arrays, units
import numpy as np

#read in dropsonde data
dropsonde = netCDF4.Dataset('D20230911_083500_PQC.nc') 

#save dropsonde data
z = dropsonde.variables['gpsalt']
z_profile_mask = z[0:]
z_profile = []
for i in range(len(z_profile_mask)):
   if z_profile_mask.mask[i] == False:
        z_profile = np.append(z_profile, z_profile_mask[i])

launch_time = dropsonde.variables['time']
launch_time_mask = launch_time[0:]

i=0
launch_time_list = []
for i in range(len(launch_time_mask)):
    if z_profile_mask.mask.data[i] == False:
        launch_time_list = np.append(launch_time_list, launch_time_mask.data[i])

#create time variable
flight_start_time = launch_time.units

YYYY = flight_start_time[14:18]
DD = flight_start_time[22:24]
MM = flight_start_time[19:21]
HH = flight_start_time[25:27]
MIN = flight_start_time[28:30]
SS = flight_start_time[31:33]

flight_start_trakfile = MM + '/' + DD + '/' +YYYY
flight_start_trakfile_time = HH + ':' + MIN + ':' + SS

pressure = dropsonde.variables['pres']
pressure_profile_mask = pressure[0:]
pressure_profile = []
for i in range(len(pressure_profile_mask)):
   if pressure_profile_mask.mask[i] == False:
        pressure_profile = np.append(pressure_profile, pressure_profile_mask[i])
        
uwind = dropsonde.variables['u_wind']
uwind_mask = uwind[0:]
uwind_profile = []
for i in range(len(uwind_mask)):
   if uwind_mask.mask[i] == False:
        uwind_profile = np.append(uwind_profile, uwind_mask[i])

vwind = dropsonde.variables['v_wind']
vwind_profile_mask = vwind[0:]
vwind_profile = []
for i in range(len(vwind_profile_mask)):
   if vwind_profile_mask.mask[i] == False:
        vwind_profile = np.append(vwind_profile, vwind_profile_mask[i])
        
dp = dropsonde.variables['dp']
dp_profile_mask = dp[0:]
dp_profile = []
for i in range(len(dp_profile_mask)):
   if dp_profile_mask.mask[i] == False:
        dp_profile = np.append(dp_profile, dp_profile_mask[i])
        
t = dropsonde.variables['tdry']
t_profile_mask = t[0:]
t_profile = []
for i in range(len(t_profile_mask)):
   if t_profile_mask.mask[i] == False:
        t_profile = np.append(t_profile, t_profile_mask[i])

#ensure data are the same same size
p = pressure_profile[0:len(dp_profile)]
T = t_profile[0:len(dp_profile)]
Td = dp_profile[0:len(dp_profile)]
u = uwind_profile[0:len(dp_profile)]
v = vwind_profile[0:len(dp_profile)]

#plot figure
fig = plt.figure(figsize=(8, 8))

# Initiate the skew-T plot type from MetPy class loaded earlier
skew = SkewT(fig, rotation=45)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
#skew.plot_barbs(p[::25], u[::25], v[::25], y_clip_radius=0.03)

# Set some appropriate axes limits for x and y
skew.ax.set_xlim(-20, 40)
skew.ax.set_ylim(1020, 100)

# Add the relevant special lines to plot throughout the figure
skew.plot_dry_adiabats(t0=np.arange(233, 533, 10) * units.K,
                       alpha=0.25, color='orangered')
skew.plot_moist_adiabats(t0=np.arange(233, 400, 5) * units.K,
                         alpha=0.25, color='tab:green')

plt.title("RFQ: "+flight_start_trakfile)
plt.savefig('Final_Project_Addington_RFQSounding.png')
plt.show()
plt.close()


#SOUNDING 4 via dropsonde (LRQ) ---------------------------------------------------------
from datetime import datetime
import os
import scipy
import metpy
import netCDF4
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import pandas_dataframe_to_unit_arrays, units
import numpy as np

#read in dropsonde data
dropsonde = netCDF4.Dataset('D20230911_055955_PQC.nc') 

#save dropsonde data
z = dropsonde.variables['gpsalt']
z_profile_mask = z[0:]
z_profile = []
for i in range(len(z_profile_mask)):
   if z_profile_mask.mask[i] == False:
        z_profile = np.append(z_profile, z_profile_mask[i])

launch_time = dropsonde.variables['time']
launch_time_mask = launch_time[0:]

i=0
launch_time_list = []
for i in range(len(launch_time_mask)):
    if z_profile_mask.mask.data[i] == False:
        launch_time_list = np.append(launch_time_list, launch_time_mask.data[i])

flight_start_time = launch_time.units

YYYY = flight_start_time[14:18]
DD = flight_start_time[22:24]
MM = flight_start_time[19:21]
HH = flight_start_time[25:27]
MIN = flight_start_time[28:30]
SS = flight_start_time[31:33]

flight_start_trakfile = MM + '/' + DD + '/' +YYYY
flight_start_trakfile_time = HH + ':' + MIN + ':' + SS

pressure = dropsonde.variables['pres']
pressure_profile_mask = pressure[0:]
pressure_profile = []
for i in range(len(pressure_profile_mask)):
   if pressure_profile_mask.mask[i] == False:
        pressure_profile = np.append(pressure_profile, pressure_profile_mask[i])
        
uwind = dropsonde.variables['u_wind']
uwind_mask = uwind[0:]
uwind_profile = []
for i in range(len(uwind_mask)):
   if uwind_mask.mask[i] == False:
        uwind_profile = np.append(uwind_profile, uwind_mask[i])

vwind = dropsonde.variables['v_wind']
vwind_profile_mask = vwind[0:]
vwind_profile = []
for i in range(len(vwind_profile_mask)):
   if vwind_profile_mask.mask[i] == False:
        vwind_profile = np.append(vwind_profile, vwind_profile_mask[i])
        
dp = dropsonde.variables['dp']
dp_profile_mask = dp[0:]
dp_profile = []
for i in range(len(dp_profile_mask)):
   if dp_profile_mask.mask[i] == False:
        dp_profile = np.append(dp_profile, dp_profile_mask[i])
        
t = dropsonde.variables['tdry']
t_profile_mask = t[0:]
t_profile = []
for i in range(len(t_profile_mask)):
   if t_profile_mask.mask[i] == False:
        t_profile = np.append(t_profile, t_profile_mask[i])

#ensure data is the same same size
p = pressure_profile[0:len(dp_profile)]
T = t_profile[0:len(dp_profile)]
Td = dp_profile[0:len(dp_profile)]
u = uwind_profile[0:len(dp_profile)]
v = vwind_profile[0:len(dp_profile)]

#plot figure
fig = plt.figure(figsize=(8, 8))

# Initiate the skew-T plot type from MetPy class loaded earlier
skew = SkewT(fig, rotation=45)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
#skew.plot_barbs(p[::25], u[::25], v[::25], y_clip_radius=0.03)

# Set some appropriate axes limits for x and y
skew.ax.set_xlim(-20, 40)
skew.ax.set_ylim(1020, 100)

# Add the relevant special lines to plot throughout the figure
skew.plot_dry_adiabats(t0=np.arange(233, 533, 10) * units.K,
                       alpha=0.25, color='orangered')
skew.plot_moist_adiabats(t0=np.arange(233, 400, 5) * units.K,
                         alpha=0.25, color='tab:green')

#add title and save
plt.title("LRQ: "+flight_start_trakfile)
plt.savefig('Final_Project_Addington_LRQSounding.png')
plt.show()
plt.close()


#METEOGRAM (Flight Level wspd and pres) --------------------------------------------------------
#import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read in data
flight_level = pd.read_csv("Flight_Level2.csv") 

time = flight_level['TIME'].tolist()
wspd = flight_level['WndSp'].tolist()
pres = flight_level['SfcPr'].tolist()
        
#create plot
fig = plt.figure(figsize=(8, 8))
       
#assign data to axes 
x = np.array(time)
y2 = np.array(pres)
y = np.array(wspd)

#allow for 2 axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

#plot data
ax1.plot(x,y, color = 'blue', label = 'wspd')
ax2.plot(x, y2, color = 'orange', label = 'pressure')

#add elements to plot
ax1.set_xlabel("time (HHMMSS)")
ax1.set_ylabel("m/s")
ax2.set_ylabel("hPa")
ax1.legend(loc = 'lower right')
ax2.legend(loc='upper right')

#add title and save
plt.title("Hurricane Lee WSPD and Pressure over time - flight I1 09/11/23")
plt.savefig('Final_Project_Addington_Timeseries.png')
plt.show()
plt.close()
 

#GRIDDED DATA/SURFACE PLOT (rh vs surface pressure) ----------------------------
import pygrib
import numpy as np
import matplotlib.pyplot as plt
import math
import cartopy.crs as ccrs
import cartopy.feature as cf

#insert GRIB file for Lee
gfs = pygrib.open("gfs_3_20230911_1200_000.grb2")

#gfs.select(name='Surface pressure')
#gfs.select(name='Relative humidity')

#read in data from grib file
rh095 = gfs[673]; rh = rh095['values'] #rh at roughly 50m AGL
pressure = gfs[561]; pres = pressure['values']

lats, lons = pressure.latlons()

#create figure
fig = plt.figure (figsize=(8,8))

#define projection
proj=ccrs.LambertConformal(central_longitude=-60.,central_latitude=10.,standard_parallels=(5.,15.))
ax=plt.axes(projection=proj)

#add features
ax.set_extent([-70.,-55.,15.,30.])
ax.add_feature(cf.LAND,color='wheat')
ax.add_feature(cf.OCEAN)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='white', alpha=0.5, linestyle='--')

gl.top_labels = False
gl.left_labels = False

#plot variables
plt.contour (lons, lats, (pres*0.01), np.arange(900,1050,10), linewidths = 2, colors='black', transform=ccrs.PlateCarree())
bounds = [80,85,90,95,100]

plt.contourf(lons, lats, rh, bounds, cmap=plt.cm.Greens, transform=ccrs.PlateCarree())

#add legend
cbar=plt.colorbar(location='bottom')
cbar.set_label ('percent')

#add title and save
plt.title ('Surface Pressure (Pa) / ~50m RH (%)')
plt.savefig('Final_Project_Addington_SfcPlot.png')
plt.show()
plt.close()



#UPPER AIR PLOT (300mb U, V WIND vs Geopotential height) -------------------------------------
#gfs.select(name='U component of wind')
#gfs.select(name='V component of wind')
#gfs.select(name='Geopotential Height')

#read in data from grib file
uwind300 = gfs[121]; uwnd = uwind300['values']
vwind300 = gfs[122]; vwnd = vwind300['values']
gpm300 = gfs[115]; gpm = gpm300['values']

lats, lons = uwind300.latlons()

#create figure
fig = plt.figure (figsize=(8,8))

#define projection
proj=ccrs.LambertConformal(central_longitude=-60.,central_latitude=10.,standard_parallels=(5.,15.))
ax=plt.axes(projection=proj)

#add features
ax.set_extent([-90.,-40.,5.,40.])
ax.add_feature(cf.LAND,color='white')
ax.add_feature(cf.OCEAN, color='lightgray')
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='white', alpha=0.5, linestyle='--')

gl.top_labels = False
gl.left_labels = False

#plot variables
bounds = [20,25,30,35,40,45,50]
wspd = np.sqrt(uwnd**2 + vwnd**2)*1.94

plt.contourf(lons,lats,wspd,bounds,cmap=plt.cm.cool,transform=ccrs.PlateCarree())

cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')

height = plt.contour(lons, lats, gpm/10, np.arange(np.min(gpm/10),np.max(gpm/10),3),linewidths = 2, colors = 'black' ,transform=ccrs.PlateCarree())

#add title and save
plt.title ('300mb Geopotential height (dm) / 300mb Wind Speed (knots)')
plt.savefig('Final_Project_Addington_UpperAirPlot.png')
plt.show()
plt.close()
