import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from myfunctions import anomalies

# === Paths ===============================================================
base_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
cases    = ["ERA5L_lpjml", "fluxcom", "nirv", "vodca2"]
start_y  = "1988"
end_y    = "2018"

unit_qnty = 10**12
gpp_units = "[TgC/grid cell]"
gpp_units_sum = "[TgC]"

#options to plot 
gpp_to_plot = "Summer" #"Summer", "all"

if gpp_to_plot == "Spring":
    sub = "$_{sp}$"
elif gpp_to_plot == "Summer":
    sub = "$_{su}$"
else: 
    sub = "$_{gs}$"



#display label for lpjml
display_labels = {"ERA5L_lpjml": "LPJmL","fluxcom": "FLUXCOM","nirv": "NIRv", "vodca2":"VODCA2"}

#thresholds
sd_threshold = -1.3
temp_th = 0.75
spi_th = -0.7

#plot fonts
font_title = 15
font_deg = 11
font_leg = 12

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

# Aridity Index classes 
AI_labels = {
    1: "Hyper Arid",    # AI < 0.03
    2: "Arid",          # 0.03 – 0.2
    3: "Semi-Arid",     # 0.2  – 0.5
    4: "Dry Sub-Humid", # 0.5  – 0.65
    5: "Humid",         # > 0.65
}


# Classification codes for SD
class_labels = {1: " dry", 2: "warm", 3: "warm and dry", 4: "other"}
class_colors = {1: "#B08968", 2: "#a3b18a", 3: "#588157", 4: "#9ca4ad"}
cat_order = list(class_labels.values())  # ["dry", "warm", "warm & dry"], consistent ordering everywhere


# === masks ===============================
#greenland mask
greenland = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks\greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0) 

#land mask
ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"]
ocean_mask = ocean_mask.isel(time=0)
land_mask = (ocean_mask == 1) 

print(land_mask.sum())

threshold_mask_lpjml = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"] == 1
threshold_mask_era5l = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"] == 1

#combining masks to get the valid land area
valid_land_base = (land_mask & greenland & threshold_mask_lpjml & threshold_mask_era5l).copy(deep=True)

#eco mask and ai mask 
eco_map  = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/Terrestrial Ecoregions of the World/data/commondata/data0/wwf_ecoregions_0.5deg_remap.nc")["ecoregion"]
eco_ids  = np.unique(eco_map.values)
eco_ids  = eco_ids[~np.isnan(eco_ids)].astype(int)
eco_ids  = eco_ids[eco_ids != 0]
eco_ids  = np.array([e for e in eco_ids if e not in SKIP_ECOS])
eco_map = eco_map.sel(lat=eco_map.lat >= 25).reindex_like(valid_land_base, method="nearest", tolerance=1e-6)

ai_map  = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\Global-AI_ET0_annual_v3\Global-AI_ET0_v3_annual/aridity_index_class.nc")["aridity_index"]
ai_ids = np.unique(ai_map.values)
ai_ids = ai_ids[~np.isnan(ai_ids)].astype(int)
ai_ids = ai_ids[ai_ids != 0]
ai_map  = ai_map.sel(lat=ai_map.lat >= 25).reindex_like(valid_land_base, method="nearest", tolerance=1e-6)

# === storage across cases ===
classification_store = {}
count_by_year_store = {}
diff_data_total = {}                                  # (lat,lon) per case: SD-vs-noSD, all SD years combined
diff_data_by_cat = {label: {} for label in cat_order}  # (lat,lon) per case per category: category mean vs noSD
diff_data_by_cat_weighted = {label: {} for label in cat_order}  # same, weighted by category's share of SD years

for case in cases:

    # === Load data ====================================================================================
    if case == "ERA5L_lpjml":
        SWEI           = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month      = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        temp = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_LPJmL_outputs/temp.nc")["temp"]
        spi = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_LPJmL_outputs/spi_pearson_03_lp5.nc")["spi_pearson_03"]
        gpp_anom       = xr.open_dataarray(base_dir + "LPJmL_gpp_anom.nc")
        
    else:
        SWEI           = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month      = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        temp = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/LPJmL_Forcing/LPJmL_Forcing_correct_units/LPJmL_Forcing_correct_units/ERA5Land_2mTemp_remapnn_filled_final.nc")["t2m"]
        spi = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/era5l_prec_spi_pearson_03.nc")["spi_pearson_03"]
        gpp_anom       = xr.open_dataarray(base_dir + case + "_gpp_anom.nc")
   
    
    #peak-month exclusion
    swe_month_valid = swe_month.where(valid_land_base)
    peak_month_ok   = (swe_month_valid <= 9) | swe_month_valid.isnull() #mask
    peak_month_ok   = peak_month_ok.reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
    valid_land_case = valid_land_base & peak_month_ok
    n_excluded = int((valid_land_base & ~peak_month_ok).sum())
    n_baseline = int(valid_land_base.sum())
    print(f"[{case}] excluding {n_excluded}/{n_baseline} pixels "
          f"({100*n_excluded/n_baseline:.2f}%) with non-physical peak SWE month (>9)")
    
    
    #===SWEI analysis period===================================================
    SWEI = SWEI.sel(time=slice(str(start_y), str(end_y))).where(valid_land_case)
    SWEI = SWEI.rename({"time": "year"})
    
    #snow droughts
    sd_mask = SWEI <= sd_threshold
    total_sd = int(sd_mask.sum())
    print("Total snow drought pixel-years:", total_sd)
    
    #===GPP====================================================================
    gpp_anom = gpp_anom.sel(time=slice(start_y, end_y)).where(valid_land_case)
    gpp_anom = gpp_anom / unit_qnty
    
    #Growing season mask 
    start_m = (swe_month + 1).clip(max=9)
    end_m   = (swe_month + 3).clip(max=9)
    
    spring_ok = xr.apply_ufunc(
         lambda m, s, e: (m >= s) & (m <= e),
         gpp_anom["time.month"], start_m, end_m,
         input_core_dims=[[], [], []],
         vectorize=True, dask="parallelized", output_dtypes=[bool], )

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
        gpp_input = gpp_anom.where(gpp_anom["time.month"].isin([6, 7, 8, 9]) & (~spring_ok),drop=True)
        print("plotting all Summer anomalies (6, 7, 8, 9) excluding months already counted in spring")

    gpp_yearly = gpp_input.groupby("time.year").sum(dim="time", skipna=True, min_count=1)
    

 
    #===temp===================================================================
    temp = temp.resample(time="MS").mean() #resampling to monthly mean
    temp_anom = anomalies(temp)
    temp_anom = temp_anom.sel(lat=temp.lat >= 25).sel(time=slice("1987-11", str(end_y))).where(valid_land_case)
    temp_3month = temp_anom.rolling(time=3,min_periods=3).mean()
    temp_3month = temp_3month.sel(time=slice(str(start_y), str(end_y)))
    
    temp_th_perc = temp_3month.quantile(temp_th, dim="time")
    
    # Select peak SWE month and reduce to one timestep per year
    temp_peak = temp_3month.where(temp_3month.time.dt.month == swe_month).groupby("time.year").max("time", skipna=True).transpose("lat", "lon","year")
    
    #===spi====================================================================
    spi = spi.sel(lat=spi.lat >= 25).sel(time=slice(str(start_y), str(end_y))).where(valid_land_case)
    spi_peak = (spi.where(spi.time.dt.month == swe_month).groupby("time.year").max("time", skipna=True))
    

    #===classifications========================================================
    valid_classification = (sd_mask & temp_peak.notnull() & spi_peak.notnull())
    warm = temp_peak > temp_th_perc
    dry  = spi_peak < spi_th

    classification = xr.full_like(SWEI, 0, dtype=np.int8)
    classification = xr.where(valid_classification & dry & ~warm,  1, classification)  # dry only (cold & dry)
    classification = xr.where(valid_classification & warm & dry,   2, classification)  # warm & dry
    classification = xr.where(valid_classification & warm & ~dry,  3, classification)  # warm only (warm & wet)
    classification = xr.where(valid_classification & ~(dry & ~warm) & ~(warm & dry) & ~(warm & ~dry),4,classification) #everything else
    classification_store[case] = classification
    
    #total SD diff (all SD years combined, not split by category)
    sd_sum   = gpp_yearly.where(sd_mask).sum(dim="year", skipna=True, min_count=1)
    sd_count = sd_mask.sum(dim="year")
    sd_mean  = sd_sum.where(sd_count > 0) / sd_count.where(sd_count > 0)

    #non-SD years mean GPP
    nosd_sum   = gpp_yearly.where(~sd_mask).sum(dim="year", skipna=True, min_count=1)
    nosd_count = (~sd_mask).sum(dim="year")
    nosd_mean  = nosd_sum.where(nosd_count > 0) / nosd_count.where(nosd_count > 0)  # (lat, lon)
        
    diff_data_total[case] = sd_mean - nosd_mean
    
    # --- per-category diffs: unweighted (effect size) + weighted (contribution)
    for code, label in class_labels.items():
        cat_mask  = (classification == code)                                        
        cat_sum   = gpp_yearly.where(cat_mask).sum(dim="year", skipna=True, min_count=1)
        cat_count = cat_mask.sum(dim="year") # n qualifying years per pixel
        cat_mean  = cat_sum.where(cat_count > 0) / cat_count.where(cat_count > 0)     
        
        diff = cat_mean - nosd_mean
        diff_data_by_cat[label][case] = diff
        
        # weight = this category's share of the pixel's SD years (dimensionless, 0-1)
        weight = cat_count / sd_count.where(sd_count > 0)          # fraction of SD years that were this type
        weighted_diff = xr.where(cat_count > 0, weight * diff, 0.0)  # 0 contribution where this type never occurred
        weighted_diff = weighted_diff.where(sd_count > 0)             # but still NaN where there's no SD at all
        diff_data_by_cat_weighted[label][case] = weighted_diff
    
    
    class_counts = {label: int((classification == code).sum()) for code, label in class_labels.items()}
    total_classified =  sum(class_counts.values())
      
    excluded_sd = total_sd - total_classified
    excluded_perc = (excluded_sd / total_sd) * 100
        
    print("Total snow drought pixel-years:", total_sd)
    print(class_counts)
    print("Classified total:", total_classified)
    print("excluded snow droughts:",  excluded_sd, excluded_perc)
     
    #per-year percentages, to plot as time series ===
    years = classification.year.values
    count_by_year = {code: [] for code in class_labels}
    
    for y in years:
        cls_y = classification.sel(year=y)
        counts_y = {code: int((cls_y == code).sum()) for code in class_labels}
        for code in class_labels:
            count_by_year[code].append(counts_y[code])
    
    count_by_year_store[case] = count_by_year
    
# =============================================================================
#dataset aggregation (mean + 3/4-agreement filter)
# ============================================================================= 
diff_stack_total = xr.concat([diff_data_total[c] for c in cases], dim="case")
diff_mean_total   = diff_stack_total.mean(dim="case", skipna=True)
n_negative_total  = (diff_stack_total < 0).sum(dim="case")
n_positive_total  = (diff_stack_total > 0).sum(dim="case")
diff_input_total  = diff_mean_total.where((n_negative_total >= 3) | (n_positive_total >= 3))
total_agree_mask = ~np.isnan(diff_input_total.values)


cum_total_by_case = {}
for c in cases:
    v = diff_data_total[c].values[total_agree_mask]
    cum_total_by_case[c] = float(np.nansum(v))
cum_total_mean = float(np.nansum(diff_input_total.values))
print("Total SD:", cum_total_by_case, "mean:", cum_total_mean)
 

#per-category effect (each category filtered independently)
diff_input_by_cat = {}
for label in cat_order:
    diff_stack = xr.concat([diff_data_by_cat[label][c] for c in cases], dim="case")
    diff_mean  = diff_stack.mean(dim="case", skipna=True)

    n_neg= (diff_stack < 0).sum(dim="case")
    n_pos = (diff_stack > 0).sum(dim="case")
    # Keep only pixels where at least 3/4 datasets agree on the sign
    diff_input_by_cat[label] = diff_mean.where((n_neg >= 3) | (n_pos >= 3))

#per-category weighted contribution, using the SAME mask as "total" (so it's additive) ===
cum_weighted_by_cat_case = {label: {} for label in cat_order}
cum_weighted_by_cat_mean = {} # mean-across-datasets version, also using total_agree_mask

for label in cat_order:
    for c in cases:
        v = diff_data_by_cat_weighted[label][c].values
        cum_weighted_by_cat_case[label][c] = float(np.nansum(v[total_agree_mask]))

    stack = xr.concat([diff_data_by_cat_weighted[label][c] for c in cases], dim="case")
    stack_mean = stack.mean(dim="case", skipna=True)
    cum_weighted_by_cat_mean[label] = float(np.nansum(stack_mean.values[total_agree_mask]))

# check: category contributions (mean) should now sum to cum_total_mean
print("Sum of weighted category means:", sum(cum_weighted_by_cat_mean.values()))

'''
#======================================
#Plots with trend for each classification
#======================================
from scipy.stats import linregress

# Plot: count of each SD type over time, stacked bars per year ==========================
for case in cases:
    years = classification_store[case].year.values
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom_actual = np.zeros(len(years))
    bottom_trend = np.zeros(len(years))
    trend_lines = []
    temp_driven_vals = np.zeros(len(years))
    for code, label in class_labels.items():
        vals = np.array(count_by_year_store[case][code])
        ax.bar(years, vals, bottom=bottom_actual, label=label, color=class_colors[code], width=0.8)

        slope, intercept, r_value, p_value, std_err = linregress(years, vals)
        trend_vals = intercept + slope * years

        if "unclassified" not in label.lower():
            ax.plot(years, bottom_trend + trend_vals, color=class_colors[code], linestyle="--", linewidth=1.8)

        # Only report "dry"-type classes individually; skip "warm" (folded into Temp driven) and "unclassified"
        if "warm" in label.lower():
            temp_driven_vals += vals
        elif "other" in label.lower():
            pass
        else:
            trend_lines.append(f"{label}: {slope:+.2f}/yr  p={p_value:.3f}  R²={r_value**2:.2f}")

        bottom_actual += vals
        bottom_trend += trend_vals

    # Combined trend for "warm" + "warm and dry" (temp driven), calculation-only, not a separate bar
    slope_td, intercept_td, r_td, p_td, se_td = linregress(years, temp_driven_vals)
    trend_lines.append(f"Temp driven (warm+warm&dry): {slope_td:+.2f}/yr  p={p_td:.3f}  R²={r_td**2:.2f}")

    slope_tot, intercept_tot, r_tot, p_tot, se_tot = linregress(years, bottom_actual)
    trend_tot = intercept_tot + slope_tot * years
    ax.plot(years, trend_tot, color="black", linestyle="-", linewidth=2.5, zorder=10)
    trend_lines.append(f"Total: {slope_tot:+.2f}/yr  p={p_tot:.3f}  R²={r_tot**2:.2f}")

    stats_text = "\n".join(trend_lines)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=font_deg,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))

    ax.set_xlabel("Year", fontsize=font_leg)
    ax.set_ylabel("Number of classified SD grid cells", fontsize=font_leg)
   
    ax.legend(fontsize=font_deg)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis='both', labelsize=font_leg)
    plt.tight_layout()
    plt.show()
'''   
    
    
# === Plot: ecoregion breakdown, 3 bars (one per category) per ecoregion ===
eco_data = {label: {} for label in cat_order}
for label in cat_order:
    for eco_id in eco_ids:
        emask = (eco_map == eco_id)
        vals = diff_input_by_cat[label].where(emask).values
        vals = vals[~np.isnan(vals)]
        eco_data[label][eco_id] = np.mean(vals) if len(vals) > 0 else np.nan

# order ecoregions by the "dry" category's mean, just to have a consistent sort
sort_ref = cat_order[0]
eco_ids_sorted = sorted(eco_ids, key=lambda e: (eco_data[sort_ref][e] if not np.isnan(eco_data[sort_ref][e]) else 0))
eco_names_sorted = [ecolabels.get(e, f"Eco {e}") for e in eco_ids_sorted]

y = np.arange(len(eco_ids_sorted))
n_cat = len(cat_order)
bar_h2 = 0.8 / n_cat

fig, ax = plt.subplots(figsize=(7, 6))

for j, label in enumerate(cat_order):
    color = class_colors[list(class_labels.keys())[list(class_labels.values()).index(label)]]
    vals = [eco_data[label][e] for e in eco_ids_sorted]
    offset = (j - (n_cat - 1) / 2) * bar_h2
    ax.barh(y + offset, vals, height=bar_h2, color=color, alpha=0.85, label=label, edgecolor='k', linewidth=0.3)

ax.set_yticks(y)
ax.set_yticklabels(eco_names_sorted, fontsize=font_deg)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel(f"∆GPP{sub} {gpp_units}", fontsize=font_leg)
ax.xaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=font_deg)
ax.tick_params(axis='x', labelsize=font_deg)
plt.tight_layout()
plt.show()


# === Plot: aridity index breakdown, 3 bars (one per category) per class ===
ai_data = {label: {} for label in cat_order}
for label in cat_order:
    for ai_id in ai_ids:
        amask = (ai_map == ai_id)
        vals = diff_input_by_cat[label].where(amask).values
        vals = vals[~np.isnan(vals)]
        ai_data[label][ai_id] = np.mean(vals) if len(vals) > 0 else np.nan

ai_ids_sorted = sorted(ai_ids, key=lambda a: (ai_data[sort_ref][a] if not np.isnan(ai_data[sort_ref][a]) else 0))
ai_names_sorted = [AI_labels.get(a, f"AI {a}") for a in ai_ids_sorted]

fig, ax = plt.subplots(figsize=(7, 6))
y = np.arange(len(ai_ids_sorted))

for j, label in enumerate(cat_order):
    color = class_colors[list(class_labels.keys())[list(class_labels.values()).index(label)]]
    vals = [ai_data[label][a] for a in ai_ids_sorted]
    offset = (j - (n_cat - 1) / 2) * bar_h2
    ax.barh(y + offset, vals, height=bar_h2, color=color, alpha=0.85, label=label, edgecolor='k', linewidth=0.3)

ax.set_yticks(y)
ax.set_yticklabels(ai_names_sorted, fontsize=font_deg)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel(f"∆GPP{sub} {gpp_units}", fontsize=font_leg)
ax.xaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=font_deg)
ax.tick_params(axis='x', labelsize=font_deg)
plt.tight_layout()
plt.show()


'''
# === Step 1: compute per-case AND mean stats for each category =============
stats_by_cat = {}  # stats_by_cat[label][row_key] = {"neg_pct", "pos_pct", "cum"}

for label in cat_order:
    diff_mean_cat = diff_input_by_cat[label]
    mean_vals = diff_mean_cat.values
    mean_vals = mean_vals[~np.isnan(mean_vals)]
    n_neg = int(np.sum(mean_vals < 0))
    n_pos = int(np.sum(mean_vals > 0))
    n_total = n_neg + n_pos

    cat_stats = {
        "mean": {
            "neg_pct": 100 * n_neg / n_total if n_total > 0 else np.nan,
            "pos_pct": 100 * n_pos / n_total if n_total > 0 else np.nan,
            "cum": float(np.nansum(mean_vals)),
        }
    }

    for c in cases:
        vals = diff_data_by_cat[label][c].values
        vals = vals[~np.isnan(vals)]
        nn  = int(np.sum(vals < 0))
        np_ = int(np.sum(vals > 0))
        nt  = nn + np_
        cat_stats[c] = {
            "neg_pct": 100 * nn / nt if nt > 0 else np.nan,
            "pos_pct": 100 * np_ / nt if nt > 0 else np.nan,
            "cum": float(np.nansum(vals)),
        }

    stats_by_cat[label] = cat_stats


# === Layout: y-axis groups = datasets (mean first, then each case), sub-bars = categories
row_keys    = ["mean"] + cases
row_display = {"mean": "mean", **{c: display_labels.get(c, c) for c in cases}}
cat_colors  = {label: class_colors[code] for code, label in class_labels.items()}

y = np.arange(len(row_keys))
n_sub = len(cat_order)
bar_h = 0.8 / n_sub
divider_y = 0.5  # midpoint between "mean" (index 0) and first dataset (index 1)


# === Plot 1: % negative / positive, grouped by dataset===
fig, ax = plt.subplots(figsize=(7, 5))

for j, label in enumerate(cat_order):
    neg_vals = [stats_by_cat[label][rk]["neg_pct"] for rk in row_keys]
    pos_vals = [stats_by_cat[label][rk]["pos_pct"] for rk in row_keys]
    offset = (j - (n_sub - 1) / 2) * bar_h
    color = cat_colors[label]

    ax.barh(y + offset, [-v for v in neg_vals], height=bar_h, color=color, alpha=0.85)
    ax.barh(y + offset, pos_vals, height=bar_h, color=color, alpha=0.4)

    for i, (nv, pv) in enumerate(zip(neg_vals, pos_vals)):
        ax.text(-nv - 1.5, y[i] + offset, f"{nv:.1f}%", va="center", ha="right", fontsize=8)
        ax.text( pv + 1.5, y[i] + offset, f"{pv:.1f}%", va="center", ha="left", fontsize=8)

ax.axhline(divider_y, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels([row_display[rk] for rk in row_keys], fontsize=font_deg)
ax.invert_yaxis()   # puts "mean" (index 0) at the top
ax.set_xlabel(f"% of ∆GPP{sub} grid cells\n(neg ←|→ pos)", fontsize=font_leg)
ax.set_xlim(-105, 105)
ax.set_xticks([-100, -50, 0, 50, 100])
ax.set_xticklabels(["100", "50", "0", "50", "100"], fontsize=11)
ax.axvspan(-105, 0, alpha=0.04, color="red")
ax.axvspan(0, 105, alpha=0.04, color="blue")
ax.spines[["top", "right"]].set_visible(False)

legend_handles = [mpatches.Patch(color=cat_colors[label], label=label) for label in cat_order]
ax.legend(handles=legend_handles, fontsize=font_deg, loc="upper right", bbox_to_anchor=(1.3, 1))

plt.tight_layout()
plt.show()
'''

'''
# === Plot 2b: cumulative ∆GPP, grouped by dataset  ===
cat_order_with_total = cat_order + ["Total"]
cat_colors_with_total = {**cat_colors, "Total": "black"}

n_sub2 = len(cat_order_with_total)
bar_h3 = 0.8 / n_sub2

fig, ax = plt.subplots(figsize=(7.5, 5))

for j, label in enumerate(cat_order):
    cum_vals = [cum_weighted_by_cat_mean[label]] + [cum_weighted_by_cat_case[label][c] for c in cases]
    offset = (j - (n_sub - 1) / 2) * bar_h
    color = cat_colors[label]

    ax.barh(y + offset, cum_vals, height=bar_h, color=color, alpha=0.85)
    for i, cv in enumerate(cum_vals):
        ax.text(cv * 1.02 if cv != 0 else 0.5, y[i] + offset, f"{cv:.2f}",
                va="center", ha="left" if cv >= 0 else "right", fontsize=8)

ax.axhline(divider_y, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels([row_display[rk] for rk in row_keys], fontsize=font_deg)
ax.invert_yaxis()
ax.set_xlabel(f"Weighted contribution to Cumul. ∆GPP{sub} {gpp_units_sum}", fontsize=font_leg)
ax.spines[["top", "right"]].set_visible(False)
xlim = ax.get_xlim()
ax.axvspan(xlim[0], 0, alpha=0.04, color="red")
ax.axvspan(0, xlim[1], alpha=0.04, color="blue")

legend_handles = [mpatches.Patch(color=cat_colors[l], label=l) for l in cat_order]
ax.legend(handles=legend_handles, fontsize=font_deg, loc="upper right", bbox_to_anchor=(1.28, 1))

plt.tight_layout()
plt.show()
'''


'''

#### with error bars ####

# === Step: compute mean, median, IQR per category per ecoregion ============
eco_data = {label: {} for label in cat_order}
for label in cat_order:
    for eco_id in eco_ids:
        emask = (eco_map == eco_id)
        vals = diff_input_by_cat[label].where(emask).values
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            q25, q75 = np.percentile(vals, [25, 75])
            eco_data[label][eco_id] = {
                "mean": np.mean(vals), "median": np.median(vals),
                "q25": q25, "q75": q75,}
        else:
            eco_data[label][eco_id] = {"mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}

# order ecoregions by the first category's mean, for a consistent sort
sort_ref = cat_order[0]
eco_ids_sorted = sorted(
    eco_ids,
    key=lambda e: (eco_data[sort_ref][e]["mean"] if not np.isnan(eco_data[sort_ref][e]["mean"]) else 0)
)
eco_names_sorted = [ecolabels.get(e, f"Eco {e}") for e in eco_ids_sorted]

y_eco = np.arange(len(eco_ids_sorted))
n_cat = len(cat_order)
bar_h2 = 0.8 / n_cat

fig, ax = plt.subplots(figsize=(7, 7))

for j, label in enumerate(cat_order):
    color = cat_colors[label]
    offset = (j - (n_cat - 1) / 2) * bar_h2
    means = [eco_data[label][e]["mean"] for e in eco_ids_sorted]

    ax.barh(y_eco + offset, means, height=bar_h2, color=color, alpha=0.85,
            label=label, edgecolor='k', linewidth=0.3)

    for i, e in enumerate(eco_ids_sorted):
        d = eco_data[label][e]
        if np.isnan(d["mean"]):
            continue
        yi = y_eco[i] + offset
        ax.plot([d["q25"], d["q75"]], [yi, yi], color='black', linewidth=0.8, solid_capstyle='round')
        ax.plot([d["q25"], d["q25"]], [yi - bar_h2*0.3, yi + bar_h2*0.3], color='black', linewidth=0.8)
        ax.plot([d["q75"], d["q75"]], [yi - bar_h2*0.3, yi + bar_h2*0.3], color='black', linewidth=0.8)
        ax.scatter(d["median"], yi, color='black', marker='|', s=60, zorder=5)

ax.set_yticks(y_eco)
ax.set_yticklabels(eco_names_sorted, fontsize=font_deg)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel(f"∆GPP{sub} {gpp_units}", fontsize=font_leg)
ax.xaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=font_deg)
ax.tick_params(axis='x', labelsize=font_deg)
ax.tick_params(axis='y', labelsize=font_deg)
plt.tight_layout()
plt.show()


# === Same for aridity index =================================================
ai_data = {label: {} for label in cat_order}
for label in cat_order:
    for ai_id in ai_ids:
        amask = (ai_map == ai_id)
        vals = diff_input_by_cat[label].where(amask).values
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            q25, q75 = np.percentile(vals, [25, 75])
            ai_data[label][ai_id] = {
                "mean": np.mean(vals), "median": np.median(vals),
                "q25": q25, "q75": q75,
            }
        else:
            ai_data[label][ai_id] = {"mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}

ai_ids_sorted = sorted(
    ai_ids,
    key=lambda a: (ai_data[sort_ref][a]["mean"] if not np.isnan(ai_data[sort_ref][a]["mean"]) else 0)
)
ai_names_sorted = [AI_labels.get(a, f"AI {a}") for a in ai_ids_sorted]

y_ai = np.arange(len(ai_ids_sorted))

fig, ax = plt.subplots(figsize=(7, 6))

for j, label in enumerate(cat_order):
    color = cat_colors[label]
    offset = (j - (n_cat - 1) / 2) * bar_h2
    means = [ai_data[label][a]["mean"] for a in ai_ids_sorted]

    ax.barh(y_ai + offset, means, height=bar_h2, color=color, alpha=0.85,
            label=label, edgecolor='k', linewidth=0.3)

    for i, a in enumerate(ai_ids_sorted):
        d = ai_data[label][a]
        if np.isnan(d["mean"]):
            continue
        yi = y_ai[i] + offset
        ax.plot([d["q25"], d["q75"]], [yi, yi], color='black', linewidth=0.8, solid_capstyle='round')
        ax.plot([d["q25"], d["q25"]], [yi - bar_h2*0.3, yi + bar_h2*0.3], color='black', linewidth=0.8)
        ax.plot([d["q75"], d["q75"]], [yi - bar_h2*0.3, yi + bar_h2*0.3], color='black', linewidth=0.8)
        ax.scatter(d["median"], yi, color='black', marker='|', s=60, zorder=5)

ax.set_yticks(y_ai)
ax.set_yticklabels(ai_names_sorted, fontsize=font_deg)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel(f"∆GPP{sub} {gpp_units}", fontsize=font_leg)
ax.xaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=font_deg)
ax.tick_params(axis='x', labelsize=font_deg)
ax.tick_params(axis='y', labelsize=font_deg)
plt.tight_layout()
plt.show()
'''