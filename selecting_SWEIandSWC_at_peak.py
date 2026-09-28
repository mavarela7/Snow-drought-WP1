import xarray as xr
import pandas as pd
from myfunctions import anomalies

#---------------------------------
#User parameters, paths and masks
#---------------------------------
cases = ["ERA5L_lpjml","ERA5L"] 
out_dir = "C:/Users\m.varela\Documents\Data_SnowDrought/Analysis_final/"

start_y = "1988"
end_y = "2018"

#greenland mask
greenland = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks/greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0) 

#land mask
ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"]
ocean_mask = ocean_mask.isel(time=0)
land_mask = (ocean_mask == 1) 


for case in cases:
    
    print("------------------------------------------------------------------ \n", case)
    
    #---SWEI and SWE ---
    if case == "ERA5L_lpjml":
        SWEI = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs\LPJmL_SWEI_final_method.nc")["SWEI_3month"]
        swe_peak = xr.open_dataset(out_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"] 
        swe_var = "SWE"
        threshold_mask = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks/threshold_mask.nc")[swe_var]
        threshold_mask = (threshold_mask == 1)
        
      
    else:
        SWEI = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_files\ERA5L_SWEI_final_method.nc")["SWEI_3month"]
        swe_peak = xr.open_dataset(out_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        swe_var = "sd"
        threshold_mask = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_files\ERA5L_snow_threshold_mask.nc")[swe_var]
        threshold_mask = (threshold_mask == 1)
        
        
    peak_broadcast = swe_peak.broadcast_like(SWEI)
    valid_land = (land_mask & greenland & threshold_mask)
    
    SWEI_at_peak = SWEI.where(SWEI["time.month"] == peak_broadcast)
    SWEI_at_peak = SWEI_at_peak.where(valid_land)
    SWEI_at_peak = (SWEI_at_peak.groupby("time.year").first().rename({"year": "time"}))
    
    #saving a file that shows per year per gridcell the value of SWEI corresponding to the month where that gridcell SWE usually peaks (mode)
    SWEI_at_peak.to_netcdf("C:/Users\m.varela\Documents\Data_SnowDrought/Analysis_final"+case+"_SWEI_final_at_peak.nc") 
    
    
    #---SWC---
    if case == "ERA5L_lpjml":
        swc = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs/swc_volumetric.nc",decode_cf=False)["swc_vol"]
        units, reference_date = swc.time.attrs['units'].split('since')
        swc['time'] = pd.date_range(start=reference_date, periods=swc.sizes['time'], freq='MS')
        swc = swc.isel(depth=0)
    else:
        swc = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought/ERA5L_files/rootzone2_swc_era5l.nc")["rootzone_swc"]
        swc = swc.rename({"valid_time": "time"})
        
    swc= swc.sel(time=slice(start_y, end_y))
    swc = swc.sel(lat=swc.lat >= 25)
       
    swc_anom = anomalies(swc)
    if case == "ERA5L_lpjml":
        swc_anom.to_netcdf("C:/Users\m.varela\Documents\Data_SnowDrought/ERA5L_files/rootzone2_swc_era5l_anom.nc")
        
    else:
        swc_anom.to_netcdf("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs/swc_volumetric_anom.nc")
    
    swc_target_month = ((swe_peak % 12) + 1)
    swc_month_broadcast = swc_target_month.broadcast_like(swc_anom)
    
    swc_post_peak =swc_anom.where(swc_anom["time.month"] == swc_month_broadcast)
    swc_post_peak = (swc_post_peak.groupby("time.year").first().rename({"year": "time"}))
    swc_post_peak = swc_post_peak.where(valid_land)
    #swc_post_peak.to_netcdf(out_dir + case +"_SWCanom_final_post_peak.nc")



            
            
            
            
            
            
            