import xarray as xr
import rioxarray  
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap

"""
#===lpjml grid===
grid = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs/LPJmL/mgpp_grid.nc")["cell_area"]

#=== from tiff to index class nc ===
# === Load the GeoTIFF ===
ai = rioxarray.open_rasterio("C:/Users/m.varela/Documents/Data_SnowDrought/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/ai_v3_yr.tif")
ai = ai.squeeze("band", drop=True)  # remove band dimension

# === Convert to xarray DataArray with proper coords ===
ai = ai.rename({"x": "lon", "y": "lat"})

# === The values are multiplied by 10000 — divide to get real AI ===
ai = ai.where(ai != ai.attrs.get("_FillValue", -9999)) / 10000

# === Regrid to your 0.5° grid (to match your swe/gpp data) ===
target_lat = grid.lat.values
target_lon = grid.lon.values

ai_regrid = ai.interp(lat=target_lat, lon=target_lon, method="linear") #bil in cdo
ai_regrid.name = "aridity_index"

# Save for reuse
#ai_regrid.to_netcdf("C:/Users\m.varela\Documents\Data_SnowDrought\Global-AI_ET0_annual_v3\Global-AI_ET0_v3_annual/aridity_index_0.5deg.nc")

def classify_aridity(ai):
    classes = xr.full_like(ai, np.nan)
    classes = xr.where(ai < 0.03,  1, classes)  # Hyper-arid
    classes = xr.where((ai >= 0.03) & (ai < 0.2),  2, classes)  # Arid
    classes = xr.where((ai >= 0.2)  & (ai < 0.5),  3, classes)  # Semi-arid
    classes = xr.where((ai >= 0.5)  & (ai < 0.65), 4, classes)  # Dry sub-humid
    classes = xr.where(ai >= 0.65,  5, classes)  # Humid
    return classes

ai_class = classify_aridity(ai_regrid).where(land_mask)
#ai_class.to_netcdf("C:/Users\m.varela\Documents\Data_SnowDrought\Global-AI_ET0_annual_v3\Global-AI_ET0_v3_annual/aridity_index_class.nc")
"""


#land mask
ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"]
ocean_mask = ocean_mask.isel(time=0)
land_mask = (ocean_mask == 1)

#greenland mask 
greenland = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks\greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0) 

threshold_mask_lpjml = xr.open_dataset(r"C:\Users\m.varela\Documents\Data_SnowDrought\masks\threshold_mask.nc")["SWE"]
threshold_mask_lpjml = (threshold_mask_lpjml == 1)
threshold_mask_era5l = xr.open_dataset(r"C:\Users\m.varela\Documents\Data_SnowDrought\ERA5L_files\ERA5L_snow_threshold_mask.nc")["sd"]
threshold_mask_era5l = (threshold_mask_era5l == 1)

valid_mask = land_mask & greenland & threshold_mask_lpjml & threshold_mask_era5l


# Define aridity class labels and colors
aridity_labels = {
    1: "Hyper-arid",
    2: "Arid",
    3: "Semi-arid",
    4: "Dry sub-humid",
    5: "Humid"
}

# Extract the colors from the colormap for each class
colors = [
    "#8c510a",  # Hyper-arid (dark brown)
    "#bf812d",  # Arid (brown)
    "#dfc27d",  # Semi-arid (sandy)
    "#80cdc1",  # Dry sub-humid (teal)
    "#01665e"   # Humid (deep green)
]

#ecoregion colors
eco_colors = {
    4:  "#b2dfb0",   # Temp. Broadleaf — light green
    5:  "#4d7a5a",   # Temp. Conifer   — dark green
    6:  "#f5c842",   # Boreal/Taiga    — yellow
    8:  "#6b3a2a",   # Temp. Grasslands— brown
    10: "#7cfc00",   # Montane         — bright lime green
    11: "#00bcd4",   # Tundra          — cyan/turquoise
    13: "#e05a4e",   # Deserts/Xeric   — red/salmon
}

#ecoregions
ecolabels = {
    4:  "Temp. Broadleaf & Mixed Forests",
    5:  "Temp. Conifer Forests",
    6:  "Boreal/Taiga",
    8:  "Temp. Grasslands, Savannas & Shrublands",
    10: "Montane Grasslands",
    11: "Tundra",
    13: "Deserts/Xeric Shrublands"}


#not relevant ecolabels
SKIP_ECOS = {1,2,3,7,9,12,14}
    
# === opening aridity file ===
ai_class = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\Global-AI_ET0_annual_v3\Global-AI_ET0_v3_annual/aridity_index_class.nc")["aridity_index"]
ai_class = ai_class.where(valid_mask)


# === eco mask====
eco_map  = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/Terrestrial Ecoregions of the World/data/commondata/data0/wwf_ecoregions_0.5deg_remap.nc")["ecoregion"]
eco_map = eco_map.where(valid_mask)
eco_ids  = np.unique(eco_map.values)
eco_ids  = eco_ids[~np.isnan(eco_ids)].astype(int)
eco_ids  = eco_ids[eco_ids != 0]
eco_ids  = np.array([e for e in eco_ids if e not in SKIP_ECOS])

# === Build a 2D color array ===
# Assign an integer index to each kept eco_id for colormap
id_to_idx = {eco_id: i + 1 for i, eco_id in enumerate(eco_ids)}  # 0 = masked/NaN

plot_data = xr.full_like(eco_map, fill_value=0, dtype=float)

for eco_id in eco_ids:
    plot_data = plot_data.where(eco_map != eco_id, other=id_to_idx[eco_id])

plot_data = plot_data.where(plot_data != 0)  # mask non-relevant regions

# === Colormap from eco_colors ===
cmap = ListedColormap([eco_colors[eid] for eid in eco_ids])
eco_bounds = np.arange(0.5, len(eco_ids) + 1.5)
eco_norm = BoundaryNorm(eco_bounds, cmap.N)

#plot fonts
font_title = 15
font_deg = 13
font_leg = 14


# =====================
# FIGURE ARIDITY MAP
# =====================
fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": ccrs.Robinson()})
ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND,    edgecolor="black", facecolor="#EFEFDB", zorder=0)
#ax.add_feature(cfeature.OCEAN,   facecolor="lightcyan", zorder=0)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)
ax.coastlines(linewidth=0.5, zorder=2)

# Use a discrete colormap with exactly 5 levels
cmap_discrete = mcolors.ListedColormap(colors)
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = mcolors.BoundaryNorm(bounds, cmap_discrete.N)

ai_class.plot(ax=ax,cmap=cmap_discrete,norm=norm,transform=ccrs.PlateCarree(),add_colorbar=False,zorder=1)

gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
gl.bottom_labels = False
gl.right_labels  = False
gl.xlabel_style = {"size": font_deg}
gl.ylabel_style = {"size": font_deg}
ax.set_title("Aridity Index Climate Classification", fontsize=font_title, pad=10)


legend_patches = [mpatches.Patch(facecolor=colors[i], edgecolor="black", linewidth=0.5, label=f"{aridity_labels[i+1]}") for i in range(5)]

ax.legend(
    handles=legend_patches,
    #title="Aridity Class",
    #title_fontsize=10,
    fontsize=font_leg,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),  # below the plot
    ncol=5                       # 2 columns
)

plt.tight_layout()
plt.show()

# ======================
# FIGURE ECOREGIONS MAP
#=======================
fig, ax = plt.subplots(figsize=(16, 8),subplot_kw={"projection": ccrs.Robinson()})
ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())

img = ax.pcolormesh(
    eco_map.lon.values,
    eco_map.lat.values,
    plot_data.values,
    cmap=cmap,
    norm=eco_norm,
    transform=ccrs.PlateCarree(),
    shading="auto"
)

ax.add_feature(cfeature.LAND, facecolor="#EFEFDB", edgecolor="black", linewidth=0.4, zorder=0)
#ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
#ax.add_feature(cfeature.OCEAN,   facecolor="lightcyan", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
ax.add_feature(cfeature.LAKES, facecolor="white", edgecolor="black", linewidth=0.3)

gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5, linewidth=0.5)
gl.top_labels    = True
gl.bottom_labels = False
gl.right_labels  = False
gl.left_labels   = True
gl.xlabel_style = {"size": font_deg}
gl.ylabel_style = {"size": font_deg}

# === Legend ===
legend_patches = [
    mpatches.Patch(facecolor=eco_colors[eid], edgecolor="black", linewidth=0.5,
                   label=ecolabels[eid])
    for eid in eco_ids
]

ax.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.4),
    ncol=3,
    fontsize=font_leg,
    frameon=True,
    title=None
)

ax.set_title("WWF Terrestrial Ecoregions", fontsize=font_title, pad=10)

plt.tight_layout()
plt.show()



# ===============================================
#### threshold mask final
# =================================================

swe_month = xr.open_dataset(r"C:\Users\m.varela\Documents\Data_SnowDrought\Analysis_final\Peak_files\LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
swe_month = swe_month.reindex_like(valid_mask, method="nearest", tolerance=1e-6)
swe_month_valid = swe_month.where(valid_mask)
peak_month_ok   = (swe_month_valid <= 9) | swe_month_valid.isnull()

swe_month_era5l = xr.open_dataset(r"C:\Users\m.varela\Documents\Data_SnowDrought\Analysis_final\Peak_files\ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
swe_month_era5l = swe_month_era5l.reindex_like(valid_mask, method="nearest", tolerance=1e-6)
swe_month_valid_era5l = swe_month_era5l.where(valid_mask)
peak_ok_era5l   = (swe_month_valid_era5l <= 9) | swe_month_valid_era5l.isnull()

peak_ok_both = peak_month_ok & peak_ok_era5l
valid_land_final = (valid_mask & peak_ok_both).drop_vars("time", errors="ignore")

n_excluded = int((valid_mask & ~peak_ok_era5l).sum())
n_baseline = int(valid_mask.sum())
print(f"Excluding {n_excluded}/{n_baseline} pixels ({100*n_excluded/n_baseline:.2f}%) "
      f"with non-physical peak SWE month")

fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": ccrs.Robinson()})
ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())

# --- Background layers (drawn first) ---
ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
ax.add_feature(cfeature.LAND, facecolor="#EFEFDB", edgecolor="black", linewidth=0.4, zorder=1)

# --- data on top of land ---
cmap = mcolors.ListedColormap(["royalblue"])
valid_land_final.where(valid_land_final == 1).plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    add_colorbar=False,
    zorder=2,
)

# --- Vector overlays drawn last (always on top) ---
ax.add_feature(cfeature.LAKES, facecolor="white", edgecolor="black", linewidth=0.3, zorder=3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=4)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", zorder=4)


gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5, linewidth=0.5)
gl.top_labels    = True
gl.bottom_labels = False
gl.right_labels  = False
gl.left_labels   = True
gl.xlabel_style = {"size": font_deg}
gl.ylabel_style = {"size": font_deg}

#ax.set_title("Final Valid Pixel Mask (LPJmL + ERA5L thresholds, peak-month exclusion applied)")
plt.show()


# === Count grid cells per ecoregion ========================================
# Uses eco_map (already masked to valid_mask), eco_ids, and ecolabels
# from the script above.
 
eco_counts = {}
aridity_counts = {}

for eco_id in eco_ids:
    count = int((eco_map == eco_id).sum())
    eco_counts[eco_id] = count
 
# print sorted by count, descending
print("Grid cells per ecoregion:")
for eco_id, count in sorted(eco_counts.items(), key=lambda x: -x[1]):
    label = ecolabels.get(eco_id, f"Unknown ({eco_id})")
    print(f"  {label}: {count:,} cells")
 
total_cells = sum(eco_counts.values())
print(f"\nTotal valid grid cells across kept ecoregions: {total_cells:,}")


for class_id, label in aridity_labels.items():
    count = int((ai_class == class_id).sum())
    aridity_counts[class_id] = count
 
print("Grid cells per aridity class:")
for class_id, count in sorted(aridity_counts.items(), key=lambda x: -x[1]):
    label = aridity_labels[class_id]
    print(f"  {label}: {count:,} cells")
 
total_cells = sum(aridity_counts.values())
print(f"\nTotal valid grid cells across aridity classes: {total_cells:,}")

#in km2 ========================================================================
grid_area = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\ERA5L_LPJmL_outputs/LPJmL/mgpp_grid.nc")["cell_area"]
grid_area = grid_area.reindex_like(valid_mask, method="nearest", tolerance=1e-6)
# grid_area = grid_area / 1e6  # uncomment if source is in m^2, not km^2
grid_area = grid_area / 1e6 

eco_area_km2 = {eco_id: float(grid_area.where(eco_map == eco_id).sum()) for eco_id in eco_ids}
aridity_area_km2 = {class_id: float(grid_area.where(ai_class == class_id).sum()) for class_id in aridity_labels}

print("Area per ecoregion (km^2):")
for eco_id, area in sorted(eco_area_km2.items(), key=lambda x: -x[1]):
    print(f"  {ecolabels.get(eco_id, f'Unknown ({eco_id})')}: {area:,.0f} km^2")

print("\nArea per aridity class (km^2):")
for class_id, area in sorted(aridity_area_km2.items(), key=lambda x: -x[1]):
    print(f"  {aridity_labels[class_id]}: {area:,.0f} km^2")
