"""
Snow drought (SD) frequency map — LPJmL vs ERA5-Land.
 
Computes, per pixel, the fraction of years (1988-2018) classified as a
snow drought (SWEI_3month <= -1.3) for each SWE dataset, averages the two
datasets, and plots the result with a contour outlining the 75th-percentile "hotspot" threshold.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#=== Paths and config==========================================================
base_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
mask_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/masks/"

start_y,end_y = "1989", "2018"
cases = ["LPJmL", "ERA5L"] 

#plot fonts
font_title = 15
font_deg = 13
font_leg = 14

sd_threshold = -1.3
sd_count_cap = 15

# define exclusion zones as (lat_min, lat_max, lon_min, lon_max)
arctic_exclusions = {
    "above_75N":        (75,  90, -180,  180),   # everything above 75°N
    "Svalbard":         (74,  81,   10,   35),
    "Arctic_Canada":    (72,  90,  -95,  -60),
    "Franz_Josef_Land": (79,  82,   40,   65),
    "Novaya_Zemlya":    (70,  78,   50,   70),
    "Severnaya_Zemlya": (73,  82,   90,  110),
}

sd_counts = []
sd_counts_raw = {} # store individual for per-dataset max stats
sd_bool_raw = {} 


# === masks ===============================
#greenland mask
greenland = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks\greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0) 

#land mask
ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"]
ocean_mask = ocean_mask.isel(time=0)
land_mask = (ocean_mask == 1) 

threshold_mask_lpjml = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"]
threshold_mask_lpjml = (threshold_mask_lpjml == 1)
threshold_mask_era5l = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"]
threshold_mask_era5l = (threshold_mask_era5l == 1)

#combining masks to get the valid land area
valid_land = land_mask & greenland & threshold_mask_lpjml & threshold_mask_era5l

#adding the arctic exclusion
for name, (la0, la1, lo0, lo1) in arctic_exclusions.items():
    mask_zone = ((valid_land.lat >= la0) & (valid_land.lat <= la1) & (valid_land.lon >= lo0) & (valid_land.lon <= lo1))
    valid_land_arctic = valid_land.where(~mask_zone)
    
#verify: print how many valid pixels remain
n_valid_before = int(valid_land.sum())
n_valid_after  = int(valid_land_arctic.sum())
print(f"\nValid pixels before exclusion : {n_valid_before:,}")
print(f"Valid pixels after  exclusion : {n_valid_after:,}")
print(f"Pixels removed                : {n_valid_before - n_valid_after:,}")


# === SD counts per dataset ==================================================
for case in cases:
    # --- load ---
    if case == "LPJmL":
        SWEI = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
    else:
        SWEI  = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        
    SWEI = SWEI.sel(time=slice(start_y, end_y)).where(valid_land_arctic)
    
    sd = (SWEI <= sd_threshold).rename({"time": "year"})
    sd = sd.sel(lat=sd.lat >= 25).transpose("year", "lat", "lon")

    sd_count = sd.sum("year")
    sd_count = sd_count.where(sd_count <= sd_count_cap) 
    n_years = len(SWEI["time"])
    
    sd_counts.append(sd_count)
    sd_counts_raw[case] = sd_count
    sd_bool_raw[case] = sd
    
sd_count_mean = xr.concat(sd_counts, dim="dataset").mean("dataset") # mean snow-drought count across datasets

sd_count_lpjml = sd_counts_raw["LPJmL"].where(valid_land_arctic) 
sd_count_era5l = sd_counts_raw["ERA5L"].where(valid_land_arctic) 
sd_count_mean_masked = sd_count_mean.where(valid_land_arctic)  

# convert counts to fraction of years ====
sd_frac_lpjml = (sd_count_lpjml / n_years ).where(valid_land_arctic) 
sd_frac_era5  = (sd_count_era5l  / n_years).where(valid_land_arctic) 


# global mean fraction (over valid land pixels)
sd_frac_mean  = (sd_count_mean / n_years).where(valid_land_arctic) # mean fraction [0-1] per pixel
global_mean_frac = float(sd_frac_mean.mean(skipna=True))   # mean of the mean pixel values
print(f"Global mean SD frequency (land, 25–75°N): {global_mean_frac*100:.1f}%")
    

hotspot_thresh = float(np.nanpercentile(sd_frac_mean.values, 75))
hotspot_thresh_count = hotspot_thresh * n_years 
print(f"75th percentile SD frequency: {hotspot_thresh*100:.1f}%")

print("Per-dataset maximum SD frequency:")
print(f"  LPJmL  max : {float(sd_frac_lpjml.max(skipna=True))*100:.1f}%")
print(f"  ERA5L  max : {float(sd_frac_era5.max( skipna=True))*100:.1f}%")

# === Decadal breakdown (print only) ========================================
# Split the 1988-2018 record into ~10-year windows and compute the average
# number of SD years within each window, per dataset and averaged.
def get_calendar_years(year_coord):
    """Return calendar-year ints from a 'year' coordinate, whatever its dtype."""
    try:
        return year_coord.dt.year.values  # datetime64 / cftime coordinate
    except (TypeError, AttributeError):
        return year_coord.values.astype(int)  # already plain year ints/floats
 
all_years = sorted(np.unique(get_calendar_years(sd_bool_raw["LPJmL"]["year"])))
year_windows = np.array_split(all_years, 3)  # 3 windows of ~10 years each
 
print("\nSD years by ~10-year window:")
for window_years in year_windows:
    label = f"{window_years[0]}-{window_years[-1]}"
    window_n_years = len(window_years)
    window_counts = []
    print(f"  {label} ({window_n_years} yrs):")
    for case in cases:
        sd_case = sd_bool_raw[case]
        case_years = get_calendar_years(sd_case["year"])
        idx = np.where(np.isin(case_years, window_years))[0]
 
        count_window = sd_case.isel(year=idx).sum("year").where(valid_land_arctic)
        count_val = float(count_window.mean(skipna=True))
        window_counts.append(count_val)
        print(f"    {case}: {count_val:.1f} yrs")
    print(f"    mean across datasets: {np.mean(window_counts):.1f} yrs")

#----------------------------------------------
#--- plotting the mean of the 2 datasets ------
#----------------------------------------------
fig, ax = plt.subplots( figsize=(14, 7),subplot_kw={"projection": ccrs.Robinson()})

stats_txt = (
    "Number of years:\n"
    f"min = {float(sd_count_mean_masked.min(skipna=True)):.0f}\n"
    f"max = {float(sd_count_mean_masked.max(skipna=True)):.0f}\n"
    f"mean = {float(sd_count_mean_masked.mean(skipna=True)):.1f}\n"
    f"p75 = {hotspot_thresh_count:.1f} \n \n"
    f"LPJmL:\n"
    f"Mean = {float(sd_count_lpjml.mean(skipna=True)):.1f}\n"
    f"Max = {float(sd_count_lpjml.max(skipna=True)):.0f}\n \n"
    f"ERA5-Land:\n"
    f"Mean = {float(sd_count_era5l.mean(skipna=True)):.1f}\n"
    f"Max = {float(sd_count_era5l.max(skipna=True)):.0f} "
)


#stats_txt = (
#    f"min = {float(sd_frac_mean.min(skipna=True))*100:.0f} %\n"
#    f"max = {float(sd_frac_mean.max(skipna=True))*100:.0f} %\n"
#    f"mean = {global_mean_frac*100:.1f} %\n"
#   f"p75 = {hotspot_thresh*100:.1f} %\n"
#    f"\n"
#    "LPJmL:\n"
#    f"Mean = {float(sd_frac_lpjml.mean(skipna=True))*100:.1f} %\n"
#    f"Max = {float(sd_frac_lpjml.max(skipna=True))*100:.1f} %\n"
#    f"ERA5-Land:\n"
#    f"Mean = {float(sd_frac_era5.mean(skipna=True))*100:.1f} %\n"
#    f"Max = {float(sd_frac_era5.max(skipna=True))*100:.1f} %"
#)

cmap = plt.cm.Blues.copy()

m = (sd_count_mean_masked).plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmin=0,
    vmax=10,
    add_colorbar=False,
)

ax.set_title("")
ax.add_feature(cfeature.LAND,facecolor="#EFEFDB")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.add_feature(cfeature.LAKES, linewidth=0.4,facecolor="#FFFFFF")
ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())

#ax.text(0.03, 0.04,stats_txt,transform=ax.transAxes,ha="left",va="bottom",fontsize=9.5, bbox=dict(facecolor="white", alpha=0.8))
ax.text(1.03, 0.5, stats_txt, transform=ax.transAxes,ha="left", va="center", fontsize=11,bbox=dict(facecolor="white", alpha=0.8),clip_on=False,)

gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.7)
gl.bottom_labels = False
gl.right_labels  = False
gl.xlabel_style = {"size": font_deg}
gl.ylabel_style = {"size": font_deg}

# contour only pixels above 75th percentile
lons = sd_frac_mean.lon.values
lats = sd_frac_mean.lat.values
frac_np = sd_frac_mean.values  # NaN outside valid land, contour ignores them

ax.contour(
    lons, lats, frac_np,
    levels=[hotspot_thresh],
    colors="black",
    linewidths=0.8,
    transform=ccrs.PlateCarree(),
    zorder=5,
)

cb = plt.colorbar(m, orientation="horizontal", pad=0.02, fraction=0.05, aspect=70)
cb.set_label("Snow Drought year count (Mean of LPJmL and ERA5-Land)", fontsize=font_leg)
cb.ax.tick_params(labelsize=font_leg) 
#plt.tight_layout()
fig.subplots_adjust(right=0.78)
plt.show()


'''
#############################################
# Plot to see the number of SD happening per gridcell, capped at 
# max 30 to get rid of the weird pixels in the arctic
#these plots the 2 datasets separately
############################################
swe_cases = [ "LPJmL","ERA5L"]

fig, axes = plt.subplots(2, 1,figsize=(16,7),subplot_kw={"projection": ccrs.PlateCarree()})
fig.subplots_adjust(wspace=0.02,hspace=0.02) #space within the maps

for ax, case in zip(axes, swe_cases):

    if case == "LPJmL":
        SWEI = xr.open_dataset( base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
    else:
        SWEI = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]

    SWEI = SWEI.sel(time=slice(start_y, end_y)).where(valid_land)

    sd = (SWEI <= -1.3).rename({"time": "year"})
    sd = sd.sel(lat=sd.lat >= 25).transpose("year", "lat", "lon")
    sd_count = sd.sum("year")
    sd_count = sd_count.where(sd_count <= 30)
    
    
    stats_txt = (
    f"max = {np.nanmax(sd_count):.0f}\n"
    f"mean = {np.nanmean(sd_count):.2f}\n"
    f"min = {np.nanmin(sd_count):.0f}")

    cmap = plt.cm.Blues.copy()
    cmap.set_bad("white")

    m = sd_count.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0,
        vmax=8,
        add_colorbar=False,
    )

    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_title(case)
    
    ax.text(
    0.98, 0.05,
    stats_txt,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    bbox=dict(facecolor="white", alpha=0.8)
)
    
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.7)
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.bottom_labels = False
    gl.right_labels  = False
#cbar = fig.colorbar(m,ax=axes,orientation="horizontal",pad=0.08,fraction=0.05)
plt.colorbar(m, ax=axes.tolist(), orientation="horizontal", pad=0.03, fraction=0.05, aspect=70, label="N snow drought years",extend="max")
plt.show()
'''