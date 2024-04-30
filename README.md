# GR655_Addington

The first four sections of FinalProject_Addington.py use dropsonde data to create soundings for the four quadrants of a hurricane. The data input files are netCDF files downloaded from the hurricane field program website (https://www.aoml.noaa.gov/data-products/#hurricanedata). A separate script sorts the files into the correct quadrant. 

Section 5 uses flight-level data from the same source as above (https://www.aoml.noaa.gov/data-products/#hurricanedata) to plot a timeseries of flight-level wind speed and pressure. The input file is a .csv file from a single flight edited to only save time, wind speed, and surface pressure. 

Sections 6 and 7 use archived GFS data (https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast) to plot surface pressure and relative humidity (section 6) and 300 mb geopotential height and wind speed (section 7). The input files are GRIB files. These files are not uploaded here due to file size limitations.  
