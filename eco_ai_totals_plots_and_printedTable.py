import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.patches as mpatches

# === Paths ===============================================================
base_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
cases    = ["ERA5L_lpjml", "fluxcom", "nirv", "vodca2"]
start_y  = "1988"
end_y    = "2018"
sd_threshold = -1.3

#plot fonts
font_title = 15
font_deg = 11
font_leg = 12
# === options to plot =====================
gpp_to_plot = "all"     # "all" or "neg" or "Summer" or "Spring"
diff_to_plot = "3agree" # "mean" or "3agree" or "4 agree"


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
valid_land_base = (land_mask & greenland & threshold_mask_lpjml & threshold_mask_era5l).copy(deep=True)

# === Labels / colors =======
#colors of the datasets
case_colors = {
    "ERA5L_lpjml": "#4472C4",
    "fluxcom":     "#E4572E",
    "nirv":        "#2CA02C",
    "vodca2":      "#9467BD"
}

#ecoregions
ecolabels = {
    4:  "Temp. Broadleaf \n & Mixed Forests",
    5:  "Temp. Conifer Forests",
    6:  "Boreal/Taiga",
    8:  "Temp. Grasslands",
    10: "Montane Grasslands",
    11: "Tundra",
    13: "Deserts/Xeric Shrublands"}

#not relevant ecolabels
SKIP_ECOS = {1,2,3,7,9,12,14}

# Aridity Index classes (UNEP/FAO convention stored as integer in the .nc file)
# Adjust the key values if your file uses different codes
AI_labels = {
    1: "Hyper Arid",    # AI < 0.03
    2: "Arid",          # 0.03 – 0.2
    3: "Semi-Arid",     # 0.2  – 0.5
    4: "Dry Sub-Humid", # 0.5  – 0.65
    5: "Humid",         # > 0.65
}

# === eco mask and ai mask ==========================
eco_map  = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/Terrestrial Ecoregions of the World/data/commondata/data0/wwf_ecoregions_0.5deg_remap.nc")["ecoregion"]
eco_ids  = np.unique(eco_map.values)
eco_ids  = eco_ids[~np.isnan(eco_ids)].astype(int)
eco_ids  = eco_ids[eco_ids != 0]
eco_ids  = np.array([e for e in eco_ids if e not in SKIP_ECOS])


ai_map  = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\Global-AI_ET0_annual_v3\Global-AI_ET0_v3_annual/aridity_index_class.nc")["aridity_index"]
ai_ids = np.unique(ai_map.values)
ai_ids = ai_ids[~np.isnan(ai_ids)].astype(int)
ai_ids = ai_ids[ai_ids != 0]

eco_map = eco_map.sel(lat=eco_map.lat >= 25).reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
ai_map  = ai_map.sel(lat=ai_map.lat >= 25).reindex_like(valid_land_base, method="nearest", tolerance=1e-6)

diff_data = {}
sd_mean_store   = {}
nosd_mean_store = {}


for case in cases:

    # === Load data ====================================================================================
    if case == "ERA5L_lpjml":
        gpp_anom       = xr.open_dataarray(base_dir + "LPJmL_gpp_anom.nc")
        SWEI           = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month      = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        
    else:
        SWEI           = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        gpp_anom       = xr.open_dataarray(base_dir + case + "_gpp_anom.nc")
        swe_month      = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
    
    #peak-month exclusion, built fresh from the frozen baseline every iteration
    swe_month_valid = swe_month.where(valid_land_base)
    peak_month_ok   = (swe_month_valid <= 9) | swe_month_valid.isnull()
    peak_month_ok   = peak_month_ok.reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
    valid_land_case = valid_land_base & peak_month_ok

    n_excluded = int((valid_land_base & ~peak_month_ok).sum())
    n_baseline = int(valid_land_base.sum())
    print(f"[{case}] excluding {n_excluded}/{n_baseline} pixels "
          f"({100*n_excluded/n_baseline:.2f}%) with non-physical peak SWE month (>9)")
    
    # === Apply masks ==================================================================================
    gpp_anom = gpp_anom.sel(time=slice(start_y, end_y)).where(valid_land_case)
    SWEI     = SWEI.sel(time=slice(start_y, end_y)).where(valid_land_case)
    
    # === making GPP units of TgC ======
    gpp_anom = gpp_anom / 10**12
    gpp_units = "[TgC/grid cell]"
    

    # === Growing season mask ==========================================================================
    start_m = (swe_month + 1).clip(max=9)
    end_m   = (swe_month + 3).clip(max=9)
    
    spring_ok = xr.apply_ufunc(
         lambda m, s, e: (m >= s) & (m <= e),
         gpp_anom["time.month"],
         start_m,
         end_m,
         input_core_dims=[[], [], []],
         vectorize=True,
         dask="parallelized",
         output_dtypes=[bool],
     )

    if gpp_to_plot == "all":
        
        month_ok = xr.apply_ufunc(
            lambda m, start: (m >= start) & (m <= 9), 
            gpp_anom["time.month"], start_m,
            input_core_dims=[[], []],
            vectorize=True, dask="parallelized", output_dtypes=[bool],)
        
        gpp_season = gpp_anom.where(month_ok, drop=False)
        gpp_input = gpp_season     
        print("plotting all anomalies")
        
        
    elif gpp_to_plot == "Spring":
        gpp_input = gpp_anom.where(spring_ok, drop=False).where(valid_land_case)
        
    elif gpp_to_plot == "Summer":
        gpp_input = gpp_anom.where(
    gpp_anom["time.month"].isin([6, 7, 8, 9]) & (~spring_ok),drop=True)
        print("plotting all Summer anomalies (6, 7, 8, 9) excluding months already counted in spring")
          

    # === SD mask: (year, lat, lon) ====================================================================
    sd_mask_yr = (SWEI <= sd_threshold).rename({"time": "year"})  # (year, lat, lon)

    # === Per-pixel mean GPP per SD / noSD event =======================================================
    # Sum GPP over growing season per year
    gpp_yearly = gpp_input.groupby("time.year").sum(dim="time", skipna=True, min_count=1)

    # SD: sum GPP only in SD years, divide by number of SD years per pixel
    sd_sum    = gpp_yearly.where(sd_mask_yr).sum(dim="year", skipna=True, min_count=1)
    sd_count  = sd_mask_yr.sum(dim="year")                                  # n SD events per pixel
    sd_mean   = sd_sum.where(sd_count > 0) / sd_count.where(sd_count > 0)  # (lat, lon)

    # noSD: same but for non-SD years
    nosd_sum   = gpp_yearly.where(~sd_mask_yr).sum(dim="year", skipna=True, min_count=1)
    nosd_count = (~sd_mask_yr).sum(dim="year")
    nosd_mean  = nosd_sum.where(nosd_count > 0) / nosd_count.where(nosd_count > 0)  # (lat, lon)

    # Difference: negative = SD years have lower GPP (drought impact)
    diff = sd_mean - nosd_mean  # (lat, lon)
    
    diff_data[case] = diff
    sd_mean_store[case]   = sd_mean
    nosd_mean_store[case] = nosd_mean


# -----------------------------------------------------------------------------
# 1. Calculate total ΔGPP for each ecoregion and each dataset
# -----------------------------------------------------------------------------

eco_results = []

for eco_id in eco_ids:

    # Ecoregion mask
    emask = (eco_map == eco_id)

    dataset_totals = []

    for case in cases:

        # Difference map for this dataset
        diff = diff_data[case]

        # Select grid cells belonging to this ecoregion
        vals = diff.where(emask).values

        # Remove NaNs
        vals = vals[~np.isnan(vals)]

        if len(vals) > 0:
            # Total ΔGPP across the entire ecoregion
            total = np.sum(vals)
        else:
            total = np.nan

        dataset_totals.append(total)

    dataset_totals = np.array(dataset_totals, dtype=float)

    # Skip ecoregions where none of the datasets has data
    if np.all(np.isnan(dataset_totals)):
        continue

    # -------------------------------------------------------------------------
    # Statistics across the 4 datasets
    # -------------------------------------------------------------------------

    mean_total = np.nanmean(dataset_totals)

    # 25th and 75th percentiles = uncertainty whiskers
    q25 = np.nanpercentile(dataset_totals, 25)
    q75 = np.nanpercentile(dataset_totals, 75)

    # Also calculate min/max in case you want full dataset spread
    min_total = np.nanmin(dataset_totals)
    max_total = np.nanmax(dataset_totals)

    name = ecolabels.get(eco_id, f"Eco {eco_id}")

    eco_results.append({
        "eco_id": eco_id,
        "ecoregion": name,

        # individual datasets
        "ERA5L_lpjml": dataset_totals[0],
        "fluxcom":     dataset_totals[1],
        "nirv":        dataset_totals[2],
        "vodca2":      dataset_totals[3],

        # ensemble statistics
        "mean": mean_total,
        "q25": q25,
        "q75": q75,
        "min": min_total,
        "max": max_total
    })


# -----------------------------------------------------------------------------
# 2. Make dataframe
# -----------------------------------------------------------------------------

df_eco = pd.DataFrame(eco_results)

# Sort by mean total ΔGPP
df_eco = df_eco.sort_values("mean").reset_index(drop=True)

print(df_eco)


# =============================================================================
# 3. Prepare asymmetric error bars
# =============================================================================

# Error bars need to be distances from the mean,
# rather than the absolute q25/q75 values.

err_low = df_eco["mean"].values - df_eco["min"].values
err_high = df_eco["max"].values - df_eco["mean"].values

xerr = np.vstack([err_low, err_high])


# -----------------------------------------------------------------------------
# Colors based on sign of mean
# -----------------------------------------------------------------------------
colors_bar = [ "#d6604d" if v < 0 else "#4393c3"for v in df_eco["mean"]]

# =============================================================================
# 4. Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 6.5))

y = np.arange(len(df_eco))

ax.barh(
    y,
    df_eco["mean"],
    xerr=xerr,
    color=colors_bar,
    alpha=0.75,
    edgecolor="k",
    linewidth=0.5,
    error_kw={
        "ecolor": "black",
        "elinewidth": 1.0,
        "capsize": 4,
        "capthick": 1.0 })

# Ecoregion names
ax.set_yticks(y)
ax.set_yticklabels(df_eco["ecoregion"])

# Zero line
ax.axvline(0,color="black",linewidth=1)
ax.set_xlabel( f"Mean total ΔGPP[TgC]",fontsize=font_leg)
ax.xaxis.grid(True,linestyle="--",alpha=0.4)
ax.set_axisbelow(True)
ax.tick_params(axis="x",labelsize=font_deg)
ax.tick_params(axis="y", labelsize=font_deg)

plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 1. Calculate total ΔGPP for each aridity class and each dataset
# -----------------------------------------------------------------------------
ai_results = []

for ai_id in ai_ids:

    # Aridity class mask
    amask = (ai_map == ai_id)

    dataset_totals = []

    for case in cases:

        # Difference map for this dataset
        diff = diff_data[case]

        # Select grid cells belonging to this aridity class
        vals = diff.where(amask).values

        # Remove NaNs
        vals = vals[~np.isnan(vals)]

        if len(vals) > 0:
            # Total ΔGPP across the entire aridity class
            total = np.sum(vals)
        else:
            total = np.nan

        dataset_totals.append(total)

    dataset_totals = np.array(dataset_totals, dtype=float)

    # Skip classes with no data
    if np.all(np.isnan(dataset_totals)):
        continue

    # -------------------------------------------------------------------------
    # Statistics across the 4 datasets
    # -------------------------------------------------------------------------

    mean_total = np.nanmean(dataset_totals)

    # 25th and 75th percentile across the four datasets
    q25 = np.nanpercentile(dataset_totals, 25)
    q75 = np.nanpercentile(dataset_totals, 75)

    # Full range, useful if you want min-max uncertainty
    min_total = np.nanmin(dataset_totals)
    max_total = np.nanmax(dataset_totals)

    name = AI_labels.get(ai_id, f"AI {ai_id}")

    ai_results.append({
        "ai_id": ai_id,
        "aridity": name,

        # Individual datasets
        "ERA5L_lpjml": dataset_totals[0],
        "fluxcom":     dataset_totals[1],
        "nirv":        dataset_totals[2],
        "vodca2":      dataset_totals[3],

        # Ensemble statistics
        "mean": mean_total,
        "q25": q25,
        "q75": q75,
        "min": min_total,
        "max": max_total
    })


# -----------------------------------------------------------------------------
# 2. DataFrame
# -----------------------------------------------------------------------------

df_ai = pd.DataFrame(ai_results)

# Sort by mean total ΔGPP
df_ai = df_ai.sort_values("mean").reset_index(drop=True)

print(df_ai)


# =============================================================================
# 3. Error bars
# =============================================================================

# Distance from mean to Q25/Q75
err_low = df_ai["mean"].values - df_ai["min"].values
err_high = df_ai["max"].values - df_ai["mean"].values

xerr = np.vstack([
    err_low,
    err_high
])


# -----------------------------------------------------------------------------
# Colors based on sign of mean
# -----------------------------------------------------------------------------

colors_bar_ai = [
    "#d6604d" if v < 0 else "#4393c3"
    for v in df_ai["mean"]
]


# =============================================================================
# 4. Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 5.5))

y = np.arange(len(df_ai))

ax.barh(y,df_ai["mean"],xerr=xerr,color=colors_bar_ai,alpha=0.75,edgecolor="k",linewidth=0.5,error_kw={
        "ecolor": "black",
        "elinewidth": 1.0,
        "capsize": 4,
        "capthick": 1.0
    })

# Aridity labels
ax.set_yticks(y)
ax.set_yticklabels(df_ai["aridity"])

# Zero line
ax.axvline(0, color="black",linewidth=1)

# X-axis label
ax.set_xlabel( f"Mean total ΔGPP [TgC]",fontsize=font_leg)
# Grid
ax.xaxis.grid(True,linestyle="--",alpha=0.4)

ax.set_axisbelow(True)

# Tick size
ax.tick_params(axis="x",labelsize=font_deg)
ax.tick_params(axis="y",labelsize=font_deg)

plt.tight_layout()
plt.show()


