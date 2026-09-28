import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Patch
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

'''
#plotting peak month histogram=================================================


swe_month_lpjml = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
threshold_mask_lpjml = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"] == 1

swe_month_e5l      = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
threshold_mask_e5l = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"] == 1

mask_final =  threshold_mask_lpjml & threshold_mask_e5l #mask &

swe_month_lpjml = swe_month_lpjml.where(mask_final)
swe_month_e5l = swe_month_e5l.where(mask_final)

# Flatten and remove NaNs
lpjml = swe_month_lpjml.values.flatten()
lpjml = lpjml[~np.isnan(lpjml)]

era5l = swe_month_e5l.values.flatten()
era5l = era5l[~np.isnan(era5l)]


# Month bins (centered on integers)
bins = np.arange(0.5, 13.5, 1)

plt.figure(figsize=(8, 5))

plt.hist(lpjml,bins=bins,alpha=0.6,color="steelblue", edgecolor="black",label="LPJmL")
plt.hist(era5l,bins=bins,alpha=0.6,color="darkorange",edgecolor="black",label="ERA5-Land")

plt.xticks(np.arange(1, 13))
plt.xlabel("Peak SWE Month")
plt.ylabel("Grid cell count")
plt.title("Distribution of Peak SWE Month")
plt.legend()

plt.tight_layout()
plt.show()
#==============================================================================
'''
# SETTINGS
years = np.arange(1979, 2011)
months = [ 3, 4, 5]
start_m = "March"
end_m = "May"

# LOAD DATASETS
lpjml = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_LPJmL_outputs/swe.nc")
lpjml = lpjml.sel(lat=lpjml.lat >= 25)

globsnow = xr.open_dataset("D:/ra25nir/GLOBSNOW/mv_files/swe_globsnow3_1979-2018_lpjml-grid.nc")
globsnow = globsnow.sel(lat=globsnow.lat >= 25)

merra2 = xr.open_dataset("D:/ra25nir/Data_complete/merra2_lpjmlgrid.nc")
merra2 = merra2.sel(lat=merra2.lat >= 25)

gldas = xr.open_dataset("D:/ra25nir/Data_complete/gldas_lpjmlgrid.nc")
gldas = gldas.sel(lat=gldas.lat >= 25)

era5 = xr.open_dataset("D:/ra25nir/Data_complete/era5_lpjmlgrid_final.nc")
era5 = era5.sel(lat=era5.lat >= 25)

era5l = xr.open_dataset("D:/ra25nir/Data_complete/era5l_lpjmlgrid.nc")
era5l = era5l.sel(lat=era5l.lat >= 25)

threshold = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/REAL_data_analysis/ERA5L_lpjml/snow_threshold_mask.nc")["SWE"]

greenland = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0)

mask = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\paper1_code\mask_3agree.nc")["3agree"] == 1



# seasonal mean (all months averaged)
def prep_seasonal(da, time_dim, scale=1.0, fill_neg=False):
    da = da.sel({time_dim: da[time_dim].dt.year.isin(years)})
    da = da.sel({time_dim: da[time_dim].dt.month.isin(months)})
    da = da * scale
    if fill_neg:
        da = da.where(da > 0, np.nan)
    return da.mean(dim=time_dim)

# ─────────────────────────────────────────────
# COMPUTE SEASONAL MEANS
# ─────────────────────────────────────────────

lpjml_seas    = prep_seasonal(lpjml["SWE"], "time")
globsnow_seas = prep_seasonal(globsnow["swe"],"time", fill_neg=True)
globsnow_seas = globsnow_seas.where(globsnow_seas != -1, np.nan)
merra2_seas   = prep_seasonal(merra2["SNOMAS"],  "time")
gldas_seas    = prep_seasonal(gldas["SWE_inst"], "time")
era5_seas     = prep_seasonal(era5["sd"], "valid_time", scale=1000.0)
era5l_seas    = prep_seasonal(era5l["sd"],"valid_time", scale=1000.0)

lpjml_seas    = lpjml_seas.where(threshold == 1).where(greenland)# & mask)
globsnow_seas = globsnow_seas.where(threshold == 1).where(greenland)#  & mask)
merra2_seas   = merra2_seas.where(threshold == 1).where(greenland)#  & mask)
gldas_seas    = gldas_seas.where(threshold == 1).where(greenland)#  & mask)
era5_seas     = era5_seas.where(threshold == 1).where(greenland)#  & mask)
era5l_seas    = era5l_seas.where(threshold == 1).where(greenland)# & mask)

# ─────────────────────────────────────────────
# COMPUTE RANGE AND MAP
# ─────────────────────────────────────────────
all_refs = xr.concat([merra2_seas, gldas_seas, era5_seas, era5l_seas,globsnow_seas], dim="new_dim")
min_seas = all_refs.min(dim="new_dim")
max_seas = all_refs.max(dim="new_dim")

binary_map = ((lpjml_seas >= min_seas) & (lpjml_seas <= max_seas)).astype(float)
binary_map = binary_map.where(lpjml_seas.notnull())

n_agree = int(binary_map.sum().values)
n_total = int(binary_map.notnull().sum().values)
pct     = 100 * n_agree / n_total if n_total > 0 else np.nan
print(f"Seasonal agreement: {pct:.1f}%")

within_range = (lpjml_seas >= min_seas) & (lpjml_seas <= max_seas)
overestimate = lpjml_seas > max_seas   # LPJmL higher than every reference dataset
underestimate = lpjml_seas < min_seas  # LPJmL lower than every reference dataset

# 3-class map: 0 = within range, 1 = overestimate, 2 = underestimate
class_map = xr.full_like(lpjml_seas, np.nan, dtype=float)
class_map = xr.where(within_range,  0, class_map)
class_map = xr.where(overestimate,  1, class_map)
class_map = xr.where(underestimate, 2, class_map)
class_map = class_map.where(lpjml_seas.notnull())
class_map = class_map#.where(mask)

n_within = int((class_map == 0).sum())
n_over   = int((class_map == 1).sum())
n_under  = int((class_map == 2).sum())
n_total  = int(class_map.notnull().sum())

pct_within = 100 * n_within / n_total if n_total > 0 else np.nan
pct_over   = 100 * n_over   / n_total if n_total > 0 else np.nan
pct_under  = 100 * n_under  / n_total if n_total > 0 else np.nan

print(f"Seasonal agreement: {pct_within:.1f}% within range, "
      f"{pct_over:.1f}% LPJmL overestimates, {pct_under:.1f}% LPJmL underestimates")

'''
# ─────────────────────────────────────────────
# FIGURE binary map
# ─────────────────────────────────────────────
# dark blue = outside range (0), yellow = within range (1)
cmap_binary = ListedColormap(["#08306b", "#ffd700"])  # dark blue, yellow

fig, ax = plt.subplots(figsize=(16, 7),subplot_kw={"projection": ccrs.Robinson()})
ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND,    edgecolor="black", facecolor="whitesmoke", zorder=0)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)
ax.coastlines(linewidth=0.5, zorder=2)

binary_map.plot(ax=ax, cmap=cmap_binary, vmin=0, vmax=1,transform=ccrs.PlateCarree(), add_colorbar=False, zorder=1)

gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
gl.bottom_labels = False
gl.right_labels  = False

ax.legend(
    handles=[
        Patch(facecolor="#08306b", label=f"Outside range  ({100-pct:.1f}%)"),
        Patch(facecolor="#ffd700", label=f"Within range   ({pct:.1f}%)"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),   # centers it below the map
    ncol=2,                         # both items in one row
    frameon=True, fontsize=10,
    title="LPJmL vs. MERRA2 / GLDAS / ERA5 / ERA5L / GlobSnow",
    title_fontsize=9
)

plt.show()

'''
# ─────────────────────────────────────────────
# FIGURE 3 classes map
# ─────────────────────────────────────────────
# yellow = within range, red = overestimate, dark blue = underestimate
cmap_3class = ListedColormap(["#ffd700", "#08306b","#b2182b"])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, cmap_3class.N)

fig, ax = plt.subplots(figsize=(16, 7), subplot_kw={"projection": ccrs.Robinson()})
ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="whitesmoke", zorder=0)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)
ax.coastlines(linewidth=0.5, zorder=2)

class_map.plot(ax=ax, cmap=cmap_3class, norm=norm,transform=ccrs.PlateCarree(), add_colorbar=False, zorder=1)

gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
gl.bottom_labels = False
gl.right_labels  = False

ax.legend(
    handles=[
        Patch(facecolor="#ffd700", label=f"Within range  ({pct_within:.1f}%)"),
        Patch(facecolor="#08306b", label=f"LPJmL overestimates  ({pct_over:.1f}%)"),
        Patch(facecolor="#b2182b", label=f"LPJmL underestimates  ({pct_under:.1f}%)"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    frameon=True, fontsize=10,
    title="LPJmL vs. MERRA2 / GLDAS / ERA5 / ERA5L / GlobSnow ("+start_m+"-"+end_m+"Mean)",
    title_fontsize=9
)

plt.show()

######
# ─────────────────────────────────────────────
# 1. SMALL-MULTIPLE MAPS: each dataset's seasonal mean SWE, same color scale
# ─────────────────────────────────────────────
datasets_seas = {
    "LPJmL":    lpjml_seas,
    "GlobSnow": globsnow_seas,
    "MERRA2":   merra2_seas,
    "GLDAS":    gldas_seas,
    "ERA5":     era5_seas,
    "ERA5-Land":    era5l_seas,
}

# shared color scale so panels are visually comparable
all_vals = np.concatenate([da.values.flatten()[~np.isnan(da.values.flatten())] for da in datasets_seas.values()])
vmax = np.nanpercentile(all_vals, 95)  # avoid a few extreme pixels dominating the scale

# ─────────────────────────────────────────────
# 2. SPREAD MAP: range (max - min) across ALL datasets, including LPJmL
# ─────────────────────────────────────────────
all_stack = xr.concat(list(datasets_seas.values()), dim="dataset")
range_map = all_stack.max(dim="dataset") - all_stack.min(dim="dataset")
range_map = range_map.where(threshold == 1).where(greenland)

fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})
ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="whitesmoke", zorder=0)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)
ax.coastlines(linewidth=0.5, zorder=2)

vmax_range = np.nanpercentile(range_map.values, 90)
mappable = range_map.plot(ax=ax, cmap="magma_r", vmin=0, vmax=300,
                            transform=ccrs.PlateCarree(), add_colorbar=False, zorder=1)

gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
gl.bottom_labels = False
gl.right_labels  = False

cbar = plt.colorbar(mappable, orientation="horizontal", pad=0.02, fraction=0.05, aspect=70)
cbar.set_label("Range across SWE datasets (max - min) [mm]", fontsize=12)
#ax.set_title("Where do the 6 datasets disagree most on seasonal mean SWE?", fontsize=14)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# 3. BOXPLOTS: overall distribution comparison, pooling all valid pixels
# ─────────────────────────────────────────────
box_data = []
box_labels = []
for name, da in datasets_seas.items():
    vals = da.values.flatten()
    vals = vals[~np.isnan(vals)]
    box_data.append(vals)
    box_labels.append(name)

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(box_data, labels=box_labels, showfliers=False, patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor("#a6bddb")
    patch.set_alpha(0.8)

ax.set_ylabel("Seasonal mean SWE [mm]", fontsize=12)
#ax.set_title("Distribution of seasonal mean SWE across all valid pixels, by dataset", fontsize=13)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

for label, median_line in zip(box_labels, bp['medians']):
    median_value = median_line.get_ydata()[0]
    print(f"{label}: median = {median_value:.2f} mm")
