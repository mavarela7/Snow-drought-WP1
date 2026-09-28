import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ====================================
# SETTINGS
# ====================================
out_dir      = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
SWEI_peak_months = [2, 3, 4, 5]
month_names      = {2: "February", 3: "March", 4: "April", 5: "May"}
month_labels     = ["J","F","M","A","M","J","J","A","S","O","N","D"]
months = np.arange(1, 13)
start_y, end_y = "1988", "2018"

#====================================
#Masks
#====================================
#greenland mask
greenland = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks\greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0) 

#land mask
ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"]
ocean_mask = ocean_mask.isel(time=0)
land_mask = (ocean_mask == 1) 

#combining masks to get the valid land area
valid_land = (land_mask & greenland)


#==================
#==LPJmL files ====
#==================

#snow relevance mask
threshold_lpjml = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks/threshold_mask.nc")["SWE"]
threshold_lpjml = (threshold_lpjml == 1)

swe_lpjml = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs\swe.nc")["SWE"]
swe_lpjml = swe_lpjml.assign_coords(month=("time", swe_lpjml.time.dt.month.data),) #adding month and year coordinates
swe_lpjml = swe_lpjml.sel(lat=swe_lpjml.lat >= 25).where(valid_land)
swe_lpjml = swe_lpjml.where(threshold_lpjml) 
swe_lpjml = swe_lpjml.where(swe_lpjml < 1000, np.nan).sel(time=slice(start_y, end_y))


melt_lpjml = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs\melt.nc")["melt"]
melt_lpjml = melt_lpjml.assign_coords(month=("time", melt_lpjml.time.dt.month.data),) #adding month and year coordinates
melt_lpjml = melt_lpjml.sel(lat=melt_lpjml.lat >= 25).where(valid_land)
melt_lpjml = melt_lpjml.where(threshold_lpjml).sel(time=slice(start_y, end_y))

swe_month_lpjml = xr.open_dataset(out_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]

#==================
#==ERA5L files ====
#==================      
threshold_e5l = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"]
threshold_e5l = (threshold_e5l == 1)

swe_e5l = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_files/era5l_lpjmlgrid_bil.nc")["sd"]
swe_e5l = swe_e5l * 1000
swe_e5l = swe_e5l.rename({"valid_time": "time"})
swe_e5l = swe_e5l.assign_coords(month=("time", swe_e5l.time.dt.month.data),) #adding month and year coordinates
swe_e5l = swe_e5l.sel(lat=swe_e5l.lat >= 25).where(valid_land)
swe_e5l = swe_e5l.where(threshold_e5l)
swe_e5l = swe_e5l.where(swe_e5l < 1000, np.nan).sel(time=slice(start_y, end_y))

melt_e5l = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_files/era5l_lpjmlgrid_bil.nc")["smlt"]
melt_e5l = melt_e5l.rename({"valid_time": "time"})
melt_e5l = melt_e5l.assign_coords(month=("time", melt_e5l.time.dt.month.data),)
melt_e5l = melt_e5l.sel(lat=melt_e5l.lat >= 25).where(valid_land)
melt_e5l = melt_e5l.where(threshold_e5l).sel(time=slice(start_y, end_y))
days_in_month = melt_e5l['time'].dt.days_in_month
melt_e5l =melt_e5l * days_in_month * 1000 

swe_month_era5l = xr.open_dataset(out_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]

#==================
#==MERRA2 files ===
#================== 
swe_merra2 =  xr.open_dataset(r"D:/ra25nir\Data_complete/merra2_lpjmlgrid.nc")["SNOMAS"]
days_in_month = swe_merra2["time"].dt.days_in_month
swe_merra2 = swe_merra2 * days_in_month * 86400  # removing per s
swe_merra2 = swe_merra2.sel(lat=swe_merra2.lat >= 25).where(valid_land)
swe_merra2 = swe_merra2.where(swe_merra2 < 200, np.nan).sel(time=slice(start_y, end_y))

melt_merra2 = xr.open_dataset(r"D:/ra25nir\Data_complete/merra2_lpjmlgrid.nc")["SMLAND"]
melt_merra2 = melt_merra2 * days_in_month * 86400  # removing per s
melt_merra2 = melt_merra2.sel(lat=melt_merra2.lat >= 25).where(valid_land).sel(time=slice(start_y, end_y))

swe_month_merra2 = xr.open_dataset(out_dir + "/Peak_files/MERRA2_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"] 




#======================================================
# FIGURE: 2x2, one panel per snow realm
#======================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
axes = axes.flatten()

colors_swe  = {"LPJmL": "tab:blue",  "ERA5L": "tab:orange","MERRA2": "tab:red"}
colors_melt = {"LPJmL": "steelblue", "ERA5L": "darkorange","MERRA2": "tab:red"}

for ax_i, peak_m in enumerate(SWEI_peak_months):
    ax   = axes[ax_i]
    
    # each product gets its own peak-month realm mask
    mask_lpjml = (swe_month_lpjml == peak_m) # (lat, lon)
    mask_e5l = (swe_month_era5l == peak_m)
    mask_merra2 = (swe_month_merra2 == peak_m)
    
    # ── climatological monthly mean across pixels in this realm ──
    # SWE
    swe_clim_lpjml  = swe_lpjml.where(mask_lpjml).mean(dim=("lat","lon"),  skipna=True).groupby("time.month").mean("time").values
    swe_clim_e5l    = swe_e5l.where(mask_e5l).mean(dim=("lat","lon"),    skipna=True).groupby("time.month").mean("time").values
    swe_clim_merra2    = swe_merra2.where(mask_lpjml).mean(dim=("lat","lon"),    skipna=True).groupby("time.month").mean("time").values
    # MELT
    melt_clim_lpjml = melt_lpjml.where(mask_lpjml).mean(dim=("lat","lon"), skipna=True).groupby("time.month").mean("time").values
    melt_clim_e5l   = melt_e5l.where(mask_e5l).mean(dim=("lat","lon"),   skipna=True).groupby("time.month").mean("time").values
    melt_clim_merra2   = melt_merra2.where(mask_e5l).mean(dim=("lat","lon"),   skipna=True).groupby("time.month").mean("time").values

    # ── find peak months ──
    peak_swe_lpjml  = int(np.nanargmax(swe_clim_lpjml))   # 0-based
    peak_melt_lpjml = int(np.nanargmax(melt_clim_lpjml))
    peak_swe_e5l    = int(np.nanargmax(swe_clim_e5l))
    peak_melt_e5l   = int(np.nanargmax(melt_clim_e5l))
    peak_swe_merra2    = int(np.nanargmax(swe_clim_merra2))
    peak_melt_merra2  = int(np.nanargmax(melt_clim_merra2))

    # ── plot ──
    ax.plot(months, swe_clim_lpjml,  color=colors_swe["LPJmL"],  lw=2.5,label="SWE | LPJmL")
    ax.plot(months, melt_clim_lpjml, color=colors_melt["LPJmL"],  lw=2.5, ls="--", label="Melt | LPJmL")
    ax.plot(months, swe_clim_e5l,    color=colors_swe["ERA5L"],   lw=2.5,label="SWE | ERA5L")
    ax.plot(months, melt_clim_e5l,   color=colors_melt["ERA5L"],  lw=2.5, ls="--", label="Melt | ERA5L")
    ax.plot(months, swe_clim_merra2,    color=colors_swe["MERRA2"],   lw=2.5,label="SWE | MERRA2")
    ax.plot(months, melt_clim_merra2,   color=colors_melt["MERRA2"],  lw=2.5, ls="--", label="Melt | MERRA2")

    # vertical lines at SWE and melt peaks
    ax.axvline(peak_swe_lpjml  + 1, color=colors_swe["LPJmL"],  lw=1.2,alpha=0.6, ls="--")
    ax.axvline(peak_melt_lpjml + 1, color=colors_melt["LPJmL"], lw=1.2,alpha=0.6, ls="-.")

    ax.text(0.78, 0.97,
            f"SWE peak:  month {peak_swe_lpjml+1} \n"
            f"Melt peak: month {peak_melt_lpjml+1} ",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", ec="gray", alpha=0.8, pad=3))

    ax.set_title(f"Peak SWE: {month_names[peak_m]}", fontsize=11)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_ylabel("SWE/Melt [mm]", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Month", fontsize=9)

    if ax_i == 0:
        ax.legend(fontsize=8, loc="upper center")

fig.suptitle("Long-term Average SWE and snowmelt",fontsize=13)
plt.tight_layout()
plt.show()
