import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import pandas as pd
import cartopy.feature as cfeature
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from myfunctions import calculate_mode
from matplotlib.colors import ListedColormap, BoundaryNorm

#greenland mask
greenland = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks\greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0) 

mask_thresh =  xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"] == 1

# Define a custom colormap for 12 months
month_colors = [
    "navy",  # January
    "blue",  # February 
    "royalblue",  # March 
    "forestgreen",  # April 
    "limegreen",  # May 
    "lime",  # June 
    "palegreen",  # July 
    "yellow",  # August 
    "gold",  # September 
    "goldenrod",  # October 
    "darkgoldenrod",  # November 
    "sienna" ,  # December 
]

custom_cmap = ListedColormap(month_colors, name="CustomMonthColors")

#dataprods = ["LPJmL","GLDAS","Globsnow","MERRA2","ERA5","ERA5L"]
dataprods = ["LPJmL","MERRA2","ERA5","ERA5L"]
#dataprods = ["MERRA2"] #these ones have the desired time period #"LPJmL","ERA5L"

to_plot = "SWE"

start_y = "1988"
end_y = "2018"

for dataprod in dataprods:

    if to_plot == "SWE":

        if dataprod == "LPJmL":
            prod_data = xr.open_dataset(r"C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs\swe.nc")["SWE"]

        elif dataprod == "MERRA2":
            prod_data = xr.open_dataset(r"D:/ra25nir\Data_complete/merra2_lpjmlgrid.nc")["SNOMAS"]
            days_in_month = prod_data["time"].dt.days_in_month
            prod_data = prod_data * days_in_month * 86400  # removing per s


        elif dataprod == "ERA5L":
            prod_data = xr.open_dataset(r"C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_files\era5l_lpjmlgrid_bil.nc")["sd"]
            prod_data = prod_data.rename({"valid_time": "time"}) * 1000

        elif dataprod == "ERA5":
            prod_data = xr.open_dataset(r"D:\ra25nir\Data_complete\era5_lpjmlgrid_final.nc")["sd"]
            prod_data = prod_data.rename({"valid_time": "time"}) * 1000

    else:  # melt

        if dataprod == "LPJmL":
            prod_data = xr.open_dataset(r"C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs\melt.nc")["melt"]

        elif dataprod == "MERRA2":
            prod_data = xr.open_dataset(r"D:/ra25nir\Data_complete/merra2_lpjmlgrid.nc")["SMLAND"]
            days_in_month = prod_data["time"].dt.days_in_month
            prod_data = prod_data * days_in_month * 86400  # removing per s

        elif dataprod == "ERA5L":
            prod_data = xr.open_dataset(r"C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs\melt.nc")["smlt"]
            prod_data = prod_data.rename({"valid_time": "time"}) * 1000

        elif dataprod == "ERA5":
            prod_data = xr.open_dataset(r"D:\ra25nir\Data_complete\era5_lpjmlgrid_final.nc")["smlt"]
            prod_data = prod_data.rename({"valid_time": "time"}) * 1000
         
    
#    elif dataprod == "GLDAS":
#        dataset = "C:/Users/m.varela/Documents/Data_SnowDrought/GLDAS/gldas_lpjml-grid-nn.nc"
#        var_prod = "SWE_inst"
#        prod = xr.open_dataset(dataset)
#        prod_data = prod[var_prod]
    
#    elif dataprod == "Globsnow":
#        dataset = "C:/Users/m.varela/Documents/Data_SnowDrought/GLOBSNOW/mv_files/swe_globsnow3_1979-2018_lpjml-grid.nc"
#        var_prod = "swe"
#        prod = xr.open_dataset(dataset)
#        prod_data = prod[var_prod]
#        prod_data = prod_data.where(prod_data !=-1, np.nan) #replacing -1 with NaN
#        prod_data = prod_data.where(prod_data > 0, np.nan) #replacing everything below 0 with NaN


    prod_data = prod_data.sel(time=slice(start_y, end_y))
    prod_data = prod_data.where(mask_thresh & greenland)
    prod_data = prod_data.sel(lat=prod_data.lat >= 25)

    #to make time be an object
    if dataprod != "LPJmL":
        prod_data["time"] = pd.to_datetime(prod_data["time"].values)
    
    else:
        prod_data["time"] = prod_data.indexes["time"].to_datetimeindex()
        
    #the years in the dataset    
    prod_group = prod_data.groupby("time.year") 

    #saving the max value of swe and their respective month (per year). Each item in the array is an xarray with lat and lon
    years = []
    month = []
    val = []
    
    #looping over the years
    for year, group_data in prod_group:
           
        group_data["time"] = pd.DatetimeIndex(group_data.time.values).month # Convert the time dimension to months
        max_val = group_data.max(dim="time") # Find the max value for each grid cell
        max_month = group_data.idxmax(dim = "time") # Find the month corresponding to the max value
        max_month = max_month.where(~np.isnan(max_val)) #Not taking into account the pixel with nan
        print(np.unique(max_month))
        
        years.append(year)
        month.append(max_month)
        val.append(max_val)
    
    years = np.array(years)
    time = pd.to_datetime([f"{y}-07-01" for y in years])
    
    da_val = xr.concat(val,dim=xr.DataArray(time, dims="time", name="time"))
    da_val = da_val.rename("peak_"+str(to_plot)+"_value")
    da_val.attrs.update({"long_name": "Annual peak "+str(to_plot),"units": "mm", "description": "Maximum "+str(to_plot)+" value within each year and grid cell"})
    #da_val.to_netcdf("C:/Users\m.varela\Documents\Data_SnowDrought\REAL_data_analysis\code_update/"+dataprod+"_peak"+str(to_plot)+"val_filter.nc")
    
    da_month = xr.concat(month,dim=xr.DataArray(time, dims="time", name="time"))
    da_month = da_month.rename("peak_"+str(to_plot)+"_month")
    da_month.attrs.update({"long_name": "Month of annual peak "+str(to_plot),"units": "month (1–12)","description": ("Calendar month during which maximum "+str(to_plot)+" occurs in each year and grid cell")})
    #da_month.to_netcdf("C:/Users\m.varela\Documents\Data_SnowDrought\REAL_data_analysis\code_update/"+dataprod+"_peak"+str(to_plot)+"month_filter.nc")
    
    '''
    # =========================
    # MODE of peak month
    # =========================
    mode_month = calculate_mode(da_month)
    
    mode_month = xr.where(mode_month == 0, np.nan, mode_month)
    mode_month = mode_month.rename("mode_peak_"+str(to_plot)+"_month")
    mode_month.attrs["description"] = "Most frequent month of annual peak "+str(to_plot)
    mode_month.attrs["units"] = "month (1–12)"
    #mode_month.to_netcdf("C:/Users\m.varela\Documents\Data_SnowDrought\REAL_data_analysis\code_update/" + dataprod + "_max" + to_plot + "month_Mode_"+str(start_y)+"-"+str(end_y)+".nc")
    
    # =========================
    # MEDIAN of peak month
    # =========================    
    median_val = da_val.quantile(0.5, dim="time", skipna=True)
    median_val =median_val.rename("p50_peak_val")
    
    # compute stats (ignoring NaNs)
    pmin_val  = float(median_val.min().values)
    pmax_val  = float(median_val.max().values)
    pmean_val = float(median_val.mean().values)
    # Add stats box inside the plot
    stats_val = (f"Min:  {pmin_val:.2f}\n" f"Mean: {pmean_val:.2f}\n" f"Max:{pmax_val:.2f}")
    
    median_month = da_month.quantile(0.5, dim="time", skipna=True)
    median_month = xr.where(median_month == 0, np.nan, median_month)
    median_month =median_month.rename("p50_peak_month")
    median_month.attrs["description"] = "Month by which 50% of SWE peaks have occurred"
    median_month.attrs["units"] = "month"
    #median_month.to_netcdf("C:/Users\m.varela\Documents\Data_SnowDrought\REAL_data_analysis\code_update/"+dataprod+"_max" + to_plot + "month_Median_"+str(start_y)+"-"+str(end_y)+".nc")


    #peak mode month
    fig, axs = plt.subplots(1, 1, figsize=(12, 5),subplot_kw={'projection': ccrs.Robinson()}) 
    ax = axs

    mappable= mode_month.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=1,vmax=12, cmap= custom_cmap,cbar_kwargs={
            "boundaries": np.arange(0.5, 13.5, 1),  # Define boundaries for 12 colors
            "ticks": np.arange(1, 13),              # Tick positions at the center of each color
            "label": "Most Frequent Month (1 = January, 12 = December)",
            "orientation": "horizontal",             # Horizontal color bar,
            "aspect": 50})
    
    ax.coastlines()
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="none")
    ax.add_feature(cfeature.LAND, edgecolor="black")
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.bottom_labels = False
    gl.right_labels = False
    
    #plt.title("Mode of Peak "+str(to_plot)+" month \n "+dataprod, fontsize=15)
    plt.show()
    '''
    
    # ─────────────────────────────────────────────
    # HISTOGRAM: how often each month is the peak SWE month
    # ─────────────────────────────────────────────
    
    # flatten all years and gridcells into one array
    month_vals = da_month.values.flatten()
    month_vals = month_vals[~np.isnan(month_vals)]
    
    # month bins (centered on integers 1-12)
    bins = np.arange(0.5, 13.5, 1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(month_vals, bins=bins, color="steelblue", edgecolor="black")
    
    ax.set_xticks(np.arange(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Count (grid cells × years)")
    ax.set_title("Distribution of Peak "+str(to_plot)+" Month\n"+dataprod)
    
    plt.tight_layout()
    plt.show()
    
    # print percentage share per month for a quick numeric read
    for m in range(1, 13):
        count_m = np.sum(month_vals == m)
        pct_m = 100 * count_m / len(month_vals)
        print(f"Month {m:2d}: {pct_m:.1f}%")
    
    
        