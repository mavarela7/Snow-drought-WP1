from scipy.stats import ttest_rel, wilcoxon,  t as t_dist
import pingouin as pg
from scipy import stats
from cohens_d import cohens_d
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# === Paths ===
base_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
cases    = ["ERA5L_lpjml", "fluxcom", "nirv", "vodca2"]
start_y  = "1988"
end_y    = "2018"

# === Options ===
gpp_to_plot  = "all"     # "all" or "neg" or "Summer" or "Spring"
diff_to_plot = "3agree"  # "mean" or "3agree" or "4agree"

# === Masks ===
greenland = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0)

ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"]
ocean_mask = ocean_mask.isel(time=0)
land_mask  = (ocean_mask == 1)

threshold_mask_lpjml = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"]
threshold_mask_lpjml = (threshold_mask_lpjml == 1)
threshold_mask_era5l = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"]
threshold_mask_era5l = (threshold_mask_era5l == 1)

valid_land = land_mask & greenland & threshold_mask_lpjml & threshold_mask_era5l

# === Labels / colors ===
case_colors = {
    "ERA5L_lpjml": "#4472C4",
    "fluxcom":     "#E4572E",
    "nirv":        "#2CA02C",
    "vodca2":      "#9467BD"
}

ecolabels = {
    4:  "Temp. Broadleaf \n & Mixed Forests",
    5:  "Temp. Conifer Forests",
    6:  "Boreal/Taiga",
    8:  "Temp. Grasslands",
    10: "Montane Grasslands",
    11: "Tundra",
    13: "Deserts/Xeric Shrublands"
}

SKIP_ECOS = {1, 2, 3, 7, 9, 12, 14}

AI_labels = {
    1: "Hyper Arid",
    2: "Arid",
    3: "Semi-Arid",
    4: "Dry Sub-Humid",
    5: "Humid",
}

# === Eco and AI maps ===
eco_map = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/Terrestrial Ecoregions of the World/data/commondata/data0/wwf_ecoregions_0.5deg_remap.nc")["ecoregion"]
eco_ids = np.unique(eco_map.values)
eco_ids = eco_ids[~np.isnan(eco_ids)].astype(int)
eco_ids = eco_ids[eco_ids != 0]
eco_ids = np.array([e for e in eco_ids if e not in SKIP_ECOS])


ai_map  = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/aridity_index_class.nc")["aridity_index"]
ai_ids  = np.unique(ai_map.values)
ai_ids  = ai_ids[~np.isnan(ai_ids)].astype(int)
ai_ids  = ai_ids[ai_ids != 0]

eco_map = eco_map.sel(lat=eco_map.lat >= 25)
ai_map  = ai_map.sel(lat=ai_map.lat >= 25)

eco_map = eco_map.reindex_like(valid_land, method="nearest", tolerance=1e-6)
ai_map  = ai_map.reindex_like(valid_land, method="nearest", tolerance=1e-6)

# === Storage ===
diff_data       = {}
sd_mean_store   = {case: {} for case in cases}
nosd_mean_store = {case: {} for case in cases}
sd_map_store   = {case: {} for case in cases}
nosd_map_store = {case: {} for case in cases}

# === Main loop ===
for case in cases:
    print(f"\n--- {case} ---")

    # --- Load ---
    if case == "ERA5L_lpjml":
        gpp_anom  = xr.open_dataarray(base_dir + "LPJmL_gpp_anom.nc")
        SWEI      = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
    else:
        gpp_anom  = xr.open_dataarray(base_dir + case + "_gpp_anom.nc")
        SWEI      = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]

    # --- Apply masks and units ---
    gpp_anom = gpp_anom.sel(time=slice(start_y, end_y)).where(valid_land)
    gpp_anom = gpp_anom / 10**12
    #gpp_units = "[TgC]"
    SWEI     = SWEI.sel(time=slice(start_y, end_y)).where(valid_land)
    
    swe_month_valid = swe_month.where(valid_land)
    peak_month_ok   = (swe_month_valid <= 9) | swe_month_valid.isnull()
    peak_month_ok   = peak_month_ok.reindex_like(valid_land, method="nearest", tolerance=1e-6)
    
    valid_land_case = valid_land & peak_month_ok
    
    n_excluded = int((valid_land_case & ~peak_month_ok).sum())
    n_baseline = int(valid_land_case.sum())
    print(f"[{case}] excluding {n_excluded}/{n_baseline} pixels "
          f"({100*n_excluded/n_baseline:.2f}%) with non-physical peak SWE month (>9)")

    # --- SD mask ---
    sd_mask_yr = (SWEI <= -1.3).rename({"time": "year"})
    sd_mask_yr = sd_mask_yr.sel(lat=sd_mask_yr.lat >= 25)
    
    
    # --- Growing season window (used for diff_data and "Growing Season" stat test) ---
    spring_start_month = (swe_month + 1).clip(max=9)
    spring_end_month = (swe_month + 3).clip(max=9)
    
    growing_ok = xr.apply_ufunc(
        lambda m, start: (m >= start) & (m <= 9),
        gpp_anom["time.month"], spring_start_month,
        input_core_dims=[[], []],
        vectorize=True, dask="parallelized", output_dtypes=[bool])

    spring_ok = xr.apply_ufunc(
        lambda m, s, e: (m >= s) & (m <= e),
        gpp_anom["time.month"], spring_start_month, spring_end_month,
        input_core_dims=[[], [], []],
        vectorize=True, dask="parallelized", output_dtypes=[bool])

    summer_ok = gpp_anom["time.month"].isin([6, 7, 8, 9]) & (~spring_ok) #excluding anything that entered in spring

    season_windows = {
        "Growing Season": growing_ok,
        "Spring":         spring_ok,
        "Summer":         summer_ok,
    }

         
    for season_name, s_ok in season_windows.items():
        
        ## s_ok : growing:ok, spring_ok, summer_ok
        
        gpp_s        = gpp_anom.where(s_ok, drop=False).where(valid_land_case)
        gpp_s_yearly = gpp_s.groupby("time.year").sum(dim="time", skipna=True, min_count=1)
    
        sd_sum_s     = gpp_s_yearly.where(sd_mask_yr).sum(dim="year", skipna=True, min_count=1)
        sd_count_s   = sd_mask_yr.sum(dim="year")
        sd_mean_2d   = sd_sum_s.where(sd_count_s > 0) / sd_count_s.where(sd_count_s > 0)
    
        nosd_sum_s   = gpp_s_yearly.where(~sd_mask_yr).sum(dim="year", skipna=True, min_count=1)
        nosd_count_s = (~sd_mask_yr).sum(dim="year")
        nosd_mean_2d = nosd_sum_s.where(nosd_count_s > 0) / nosd_count_s.where(nosd_count_s > 0)
        
    
        # --- Store 2D maps for spatial subsetting (ecoregion/aridity tests) ---
        sd_map_store[case][season_name]   = sd_mean_2d
        nosd_map_store[case][season_name] = nosd_mean_2d
        #diff_data[case] = sd_mean_2d - nosd_mean_2d
    
        # --- Store 1D valid pairs for global tests ---
        sd_mean_s_flat   = sd_mean_2d.values.ravel()
        nosd_mean_s_flat = nosd_mean_2d.values.ravel()
        valid_pairs      = ~np.isnan(sd_mean_s_flat) & ~np.isnan(nosd_mean_s_flat)
        sd_mean_store[case][season_name]   = sd_mean_s_flat[valid_pairs]
        nosd_mean_store[case][season_name] = nosd_mean_s_flat[valid_pairs]
        print(f"  {season_name}: {valid_pairs.sum()} valid pixel pairs")

####################################
# === to test on a global level ===
####################################
test_seasons = ["Growing Season", "Spring", "Summer"]
rows = []

for case in cases:
    for season in test_seasons:
        sd_v   = sd_mean_store[case][season]
        nosd_v = nosd_mean_store[case][season]
        diff   = sd_v - nosd_v
        n      = len(diff)

        if n < 5:
            continue

        # --- Tests ---
        ttest_res = pg.ttest(sd_v, nosd_v, paired=True, confidence=0.95)

        p_t      = float(ttest_res["p-val"].values[0])
        cohens_d = float(ttest_res["cohen-d"].values[0])         # unsigned, so preserve sign manually
        ci95     = ttest_res["CI95%"].values[0]                  # [low, high]
        ci95_low, ci95_high = float(ci95[0]), float(ci95[1])
        
        # Hedges g (bias-corrected Cohen's d)
        g = pg.compute_effsize(sd_v, nosd_v, paired=True, eftype="hedges")
        
        # Sign of effect
        mean_diff = float(np.mean(diff))
        d_signed  = cohens_d * np.sign(mean_diff)
        g_signed  = g        * np.sign(mean_diff)

        # 99% CI (pingouin ttest only does 95%, compute 99% separately)
        ci99_res    = pg.ttest(sd_v, nosd_v, paired=True, confidence=0.99)
        ci99        = ci99_res["CI99%"].values[0]  
        ci99_low, ci99_high = float(ci99[0]), float(ci99[1])
        
        # --- Wilcoxon ---
        _, p_w = wilcoxon(diff, alternative="two-sided")

        # --- Effect size interpretation ---
        def interpret_d(val):
            val = abs(val)
            if val < 0.2:   return "negligible"
            elif val < 0.5: return "small"
            elif val < 0.8: return "medium"
            else:           return "large"
            
        UNIT_SCALE = 1e12  # gC -> TgC, applied only to magnitude-bearing outputs


        rows.append({
            "Case":        case,
            "Season":      season,
            "n pixels":    n,
            "mean diff":   round(mean_diff, 6),
            "std diff":    round(float(np.std(diff, ddof=1)), 6),
            "Cohen's d":   round(d_signed, 4),
            "Hedges' g":   round(g_signed, 4),
            "Effect size": interpret_d(g_signed),
            "CI95 low":    round(ci95_low, 6),
            "CI95 high":   round(ci95_high, 6),
            "CI99 low":    round(ci99_low, 6),
            "CI99 high":   round(ci99_high, 6),
            "0 in CI95":   ci95_low <= 0 <= ci95_high,
            "0 in CI99":   ci99_low <= 0 <= ci99_high,
            "t-test p":    round(p_t, 4),
            "Wilcoxon p":  round(p_w, 4),
        })
       
results_df = pd.DataFrame(rows).set_index(["Case", "Season"])

# === Print ===
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\n=== Effect size & Confidence Intervals ===")
print(results_df.to_string())

print("\n=== Key summary ===")
summary = results_df[["mean diff", "CI95 low", "CI95 high", "Hedges' g", "Effect size", "0 in CI95", "0 in CI99"]]
print(summary.to_string())


####################################
#------per ecoregion test
####################################
# === Group definitions ===
eco_order    = [13, 8, 4, 10, 5, 6, 11]
ai_order     = [1, 2, 3, 4, 5]

def run_full_stats(sd_vals, nosd_vals, min_n=10):
    valid  = ~np.isnan(sd_vals) & ~np.isnan(nosd_vals)
    sd_v   = sd_vals[valid]
    nosd_v = nosd_vals[valid]
    n      = len(sd_v)

    empty = {
        "n": n, "mean diff": np.nan, "std diff": np.nan,
        "Cohen's d": np.nan, "Hedges' g": np.nan, "Effect size": "—",
        "CI95 low": np.nan, "CI95 high": np.nan,
        "CI99 low": np.nan, "CI99 high": np.nan,
        "0 in CI95": np.nan, "0 in CI99": np.nan,
        "t-test p": np.nan, "Wilcoxon p": np.nan,
    }

    if n < min_n:
        return empty

    diff      = sd_v - nosd_v
    mean_diff = float(np.mean(diff))

    # --- Paired t-test + CI95 via pingouin ---
    ttest_95  = pg.ttest(sd_v, nosd_v, paired=True, confidence=0.95)
    ttest_99  = pg.ttest(sd_v, nosd_v, paired=True, confidence=0.99)

    p_t             = float(ttest_95["p-val"].values[0])
    ci95_low, ci95_h = ttest_95["CI95%"].values[0]
    ci99_low, ci99_h = ttest_99["CI99%"].values[0]  

    # --- Effect sizes via pingouin (signed) ---
    g = float(pg.compute_effsize(sd_v, nosd_v, paired=True,  eftype="hedges")) * np.sign(mean_diff)
    d = float(pg.compute_effsize(sd_v, nosd_v, paired=True, eftype="cohen"))  * np.sign(mean_diff)

    # --- Wilcoxon ---
    _, p_w = wilcoxon(diff, alternative="two-sided")
   
    return {
        "n":          n,
        "mean diff":  mean_diff,
        "std diff":   float(np.std(diff, ddof=1)),
        "Cohen's d":  round(d, 4),
        "Hedges' g":  round(g, 4),
        "Effect size":interpret_d(g),
        "CI95 low":   float(ci95_low),
        "CI95 high":  float(ci95_h),
        "CI99 low":   float(ci99_low),
        "CI99 high":  float(ci99_h),
        "0 in CI95":  float(ci95_low) <= 0 <= float(ci95_h),
        "0 in CI99":  float(ci99_low) <= 0 <= float(ci99_h),
        "t-test p":   round(p_t, 4),
        "Wilcoxon p": round(p_w, 4),
    }





# === Run per ecoregion ===
print("\n" + "="*80)
print("EFFECT SIZE & CI PER ECOREGION")
print("="*80)

for season in test_seasons:
    print(f"\n--- {season} ---")
    rows = []
    for case in cases:
        row = {"Case": case}
        for gid in eco_order:
            mask          = (eco_map == gid).values.ravel()
            label         = ecolabels.get(gid, str(gid))
            sd_v_eco      = np.where(mask, sd_map_store[case][season].values.ravel(),   np.nan)
            nosd_v_eco    = np.where(mask, nosd_map_store[case][season].values.ravel(), np.nan)

            res    = run_full_stats(sd_v_eco, nosd_v_eco)
            sign   = "↑" if res["mean diff"] > 0 else "↓"
            star_t = "*" if not np.isnan(res["t-test p"])   and res["t-test p"]   < 0.05 else ""
            star_w = "*" if not np.isnan(res["Wilcoxon p"]) and res["Wilcoxon p"] < 0.05 else ""
            hg, eff           = res["Hedges' g"], res["Effect size"]
            md                = res["mean diff"]          # <-- NEW
            ci95_lo, ci95_hi  = res["CI95 low"], res["CI95 high"]
            p_t, p_w, n       = res["t-test p"], res["Wilcoxon p"], res["n"]
            
            row[label] = (
                f"{sign} mean={md:+.4f}  g={hg:.2f} ({eff})\n"   # <-- mean diff added here
                f"   CI95:[{ci95_lo:.4f},{ci95_hi:.4f}]\n"
                f"   t:{p_t:.3f}{star_t} W:{p_w:.3f}{star_w} n={n}"
            )
            
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Case")
    print(df.to_string())


# === Run per aridity ===
print("\n" + "="*80)
print("EFFECT SIZE & CI PER ARIDITY CLASS")
print("="*80)

for season in test_seasons:
    print(f"\n--- {season} ---")
    rows = []
    for case in cases:
        row = {"Case": case}
        for aid in ai_order:
            mask         = (ai_map == aid).values.ravel()
            label        = AI_labels.get(aid, str(aid))
            sd_v_ai      = np.where(mask, sd_map_store[case][season].values.ravel(),   np.nan)
            nosd_v_ai    = np.where(mask, nosd_map_store[case][season].values.ravel(), np.nan)

            res    = run_full_stats(sd_v_ai, nosd_v_ai)
            sign   = "↑" if res["mean diff"] > 0 else "↓"
            star_t = "*" if not np.isnan(res["t-test p"])   and res["t-test p"]   < 0.05 else ""
            star_w = "*" if not np.isnan(res["Wilcoxon p"]) and res["Wilcoxon p"] < 0.05 else ""
            hg, eff           = res["Hedges' g"], res["Effect size"]
            md                = res["mean diff"]          # <-- NEW
            ci95_lo, ci95_hi  = res["CI95 low"], res["CI95 high"]
            p_t, p_w, n       = res["t-test p"], res["Wilcoxon p"], res["n"]
            
            row[label] = (
                f"{sign} mean={md:+.4f}  g={hg:.2f} ({eff})\n"
                f"   CI95:[{ci95_lo:.4f},{ci95_hi:.4f}]\n"
                f"   t:{p_t:.3f}{star_t} W:{p_w:.3f}{star_w} n={n}"
            )
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Case")
    print(df.to_string())


# === Clean summary table: just Hedges g and effect size ===
print("\n" + "="*80)
print("HEDGES G SUMMARY TABLE")
print("="*80)

for group_name, group_ids, group_labels, group_map_da in [
    ("Ecoregion", eco_order, [ecolabels.get(g, str(g)) for g in eco_order], eco_map),
    ("Aridity",   ai_order,  [AI_labels.get(a, str(a)) for a in ai_order],  ai_map),]:

    print(f"\n--- {group_name} ---")
    for season in test_seasons:
        print(f"\n  {season}:")
        rows = []
        for case in cases:
            row = {"Case": case}
            for gid, label in zip(group_ids, group_labels):
                mask         = (group_map_da == gid).values.ravel()
                sd_v         = np.where(mask, sd_map_store[case][season].values.ravel(),   np.nan)
                nosd_v       = np.where(mask, nosd_map_store[case][season].values.ravel(), np.nan)
                res          = run_full_stats(sd_v, nosd_v)
                
                hg   = res["Hedges' g"]
                eff  = res["Effect size"]
                md   = res["mean diff"] 
                row[label] = f"mean={md:+.4f}  {hg:.2f} ({eff})"
            rows.append(row)
        df = pd.DataFrame(rows).set_index("Case")
        print(df.to_string())


def run_combined_group(case, season, group_map, group_ids_to_combine):
    mask = np.zeros(group_map.shape, dtype=bool).ravel()
    for gid in group_ids_to_combine:
        mask |= (group_map == gid).values.ravel()

    sd_v   = np.where(mask, sd_map_store[case][season].values.ravel(),   np.nan)
    nosd_v = np.where(mask, nosd_map_store[case][season].values.ravel(), np.nan)

    return run_full_stats(sd_v, nosd_v)

combined_groups = {
    "Arid + Hyper Arid":    {"map": ai_map,  "ids": [1, 2]},   # Hyper Arid, Arid
    "Tundra + Boreal/Taiga":{"map": eco_map, "ids": [6, 11]},  # Boreal/Taiga, Tundra
}

print("\n" + "="*80)
print("COMBINED GROUPS: Hedges' g summary")
print("="*80)

for season in test_seasons:
    print(f"\n  {season}:")
    rows = []
    for case in cases:
        row = {"Case": case}
        for label, spec in combined_groups.items():
            res = run_combined_group(case, season, spec["map"], spec["ids"])
            hg, eff, n = res["Hedges' g"], res["Effect size"], res["n"]
            md = res["mean diff"]                      # <-- NEW
            row[label] = f"mean={md:+.4f}  {hg:.2f} ({eff})  n={n}"
        # delete the stray "row[label] = ..." line that was here before — it's outside the loop
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Case")
    print(df.to_string())


####################################
# === MULTI-DATASET (ENSEMBLE) MEAN: 3-of-4 sign agreement ===
####################################
# For each season:
#   1. Compute per-case pixel-wise diff = sd_mean_2d - nosd_mean_2d
#   2. Compute sign agreement across the 4 cases (>=3 of 4 agree)
#   3. Build ensemble mean SD and noSD maps (mean across cases, pixel-wise),
#      masked to only the >=3/4 agreement pixels
#   4. Run run_full_stats on the ensemble mean, at Global / Ecoregion / Aridity levels

print("\n" + "="*80)
print("ENSEMBLE MEAN (3-of-4 sign agreement) — STATS")
print("="*80)

ensemble_sd_store   = {}  # season -> 2D DataArray (ensemble mean, agreement-masked)
ensemble_nosd_store = {}
ensemble_agree_mask = {}  # season -> 2D boolean DataArray (>=3/4 agree)

eco_map = eco_map.sel(lat=eco_map.lat >= 25)
eco_map = eco_map.reindex_like(valid_land, method="nearest", tolerance=1e-6)

for season in test_seasons:

    # --- Stack SD/noSD maps across the 4 cases (preserves lat/lon dims) ---
    sd_stack   = xr.concat([sd_map_store[c][season]   for c in cases], dim="case")
    nosd_stack = xr.concat([nosd_map_store[c][season] for c in cases], dim="case")
    sd_stack   = sd_stack.assign_coords(case=cases)
    nosd_stack = nosd_stack.assign_coords(case=cases)

    diff_stack = sd_stack - nosd_stack  # dims: case, lat, lon

    # --- Sign agreement: count pixels where sign matches the per-pixel majority sign ---
    sign_stack   = xr.apply_ufunc(np.sign, diff_stack)
    pos_count    = (sign_stack > 0).sum(dim="case")
    neg_count    = (sign_stack < 0).sum(dim="case")
    n_valid_case = (~diff_stack.isnull()).sum(dim="case")  # cases with non-NaN diff at this pixel

    agree_count  = xr.where(pos_count >= neg_count, pos_count, neg_count)
    agree_mask   = (agree_count >= 3) & (n_valid_case >= 3)  # need >=3 valid cases to agree among

    ensemble_agree_mask[season] = agree_mask

    # --- Ensemble mean maps (mean across cases, pixel-wise), then masked to agreement pixels ---
    sd_ens_mean   = sd_stack.mean(dim="case", skipna=True).where(agree_mask)
    nosd_ens_mean = nosd_stack.mean(dim="case", skipna=True).where(agree_mask)

    ensemble_sd_store[season]   = sd_ens_mean
    ensemble_nosd_store[season] = nosd_ens_mean

    n_agree_pixels = int(agree_mask.sum())
    print(f"[{season}] {n_agree_pixels} pixels with >=3/4 case sign agreement")


# === Global (NH-wide) ensemble stats ===

print("\n--- ENSEMBLE MEAN: Global (NH-wide) ---")
ens_rows = []
for season in test_seasons:
    sd_flat   = ensemble_sd_store[season].values.ravel()
    nosd_flat = ensemble_nosd_store[season].values.ravel()
    res = run_full_stats(sd_flat, nosd_flat)
    res["Season"] = season
    ens_rows.append(res)

ens_global_df = pd.DataFrame(ens_rows).set_index("Season")
print(ens_global_df.to_string())


# === Per-ecoregion ensemble stats ===
print("\n" + "="*80)
print("ENSEMBLE MEAN: PER ECOREGION")
print("="*80)

for season in test_seasons:
    print(f"\n--- {season} ---")
    row = {}
    for gid in eco_order:
        mask       = (eco_map == gid).values.ravel()
        label      = ecolabels.get(gid, str(gid))
        sd_v_eco   = np.where(mask, ensemble_sd_store[season].values.ravel(),   np.nan)
        nosd_v_eco = np.where(mask, ensemble_nosd_store[season].values.ravel(), np.nan)

        res    = run_full_stats(sd_v_eco, nosd_v_eco)
        sign   = "↑" if (not np.isnan(res["mean diff"]) and res["mean diff"] > 0) else "↓"
        star_t = "*" if not np.isnan(res["t-test p"])   and res["t-test p"]   < 0.05 else ""
        star_w = "*" if not np.isnan(res["Wilcoxon p"]) and res["Wilcoxon p"] < 0.05 else ""
        hg, eff          = res["Hedges' g"], res["Effect size"]
        md               = res["mean diff"]
        ci95_lo, ci95_hi = res["CI95 low"], res["CI95 high"]
        p_t, p_w, n      = res["t-test p"], res["Wilcoxon p"], res["n"]

        row[label] = (
            f"{sign} mean={md:+.6f}  g={hg:.2f} ({eff})\n"
            f"   CI95:[{ci95_lo:.6f},{ci95_hi:.6f}]\n"
            f"   t:{p_t:.3f}{star_t} W:{p_w:.3f}{star_w} n={n}"
        )

    df = pd.DataFrame([row], index=["Ensemble mean (3/4 agree)"])
    print(df.to_string())


# === Per-aridity ensemble stats ===
print("\n" + "="*80)
print("ENSEMBLE MEAN: PER ARIDITY CLASS")
print("="*80)

for season in test_seasons:
    print(f"\n--- {season} ---")
    row = {}
    for aid in ai_order:
        mask      = (ai_map == aid).values.ravel()
        label     = AI_labels.get(aid, str(aid))
        sd_v_ai   = np.where(mask, ensemble_sd_store[season].values.ravel(),   np.nan)
        nosd_v_ai = np.where(mask, ensemble_nosd_store[season].values.ravel(), np.nan)

        res    = run_full_stats(sd_v_ai, nosd_v_ai)
        sign   = "↑" if (not np.isnan(res["mean diff"]) and res["mean diff"] > 0) else "↓"
        star_t = "*" if not np.isnan(res["t-test p"])   and res["t-test p"]   < 0.05 else ""
        star_w = "*" if not np.isnan(res["Wilcoxon p"]) and res["Wilcoxon p"] < 0.05 else ""
        hg, eff          = res["Hedges' g"], res["Effect size"]
        md               = res["mean diff"]
        ci95_lo, ci95_hi = res["CI95 low"], res["CI95 high"]
        p_t, p_w, n      = res["t-test p"], res["Wilcoxon p"], res["n"]

        row[label] = (
            f"{sign} mean={md:+.6f}  g={hg:.2f} ({eff})\n"
            f"   CI95:[{ci95_lo:.6f},{ci95_hi:.6f}]\n"
            f"   t:{p_t:.3f}{star_t} W:{p_w:.3f}{star_w} n={n}"
        )

    df = pd.DataFrame([row], index=["Ensemble mean (3/4 agree)"])
    print(df.to_string())


# === Clean summary: Hedges g only, ensemble mean ===
print("\n" + "="*80)
print("ENSEMBLE MEAN — HEDGES G SUMMARY TABLE")
print("="*80)

for group_name, group_ids, group_labels, group_map_da in [
    ("Ecoregion", eco_order, [ecolabels.get(g, str(g)) for g in eco_order], eco_map),
    ("Aridity",   ai_order,  [AI_labels.get(a, str(a)) for a in ai_order],  ai_map),]:

    print(f"\n--- {group_name} ---")
    for season in test_seasons:
        print(f"\n  {season}:")
        row = {}
        for gid, label in zip(group_ids, group_labels):
            mask       = (group_map_da == gid).values.ravel()
            sd_v       = np.where(mask, ensemble_sd_store[season].values.ravel(),   np.nan)
            nosd_v     = np.where(mask, ensemble_nosd_store[season].values.ravel(), np.nan)
            res        = run_full_stats(sd_v, nosd_v)

            hg  = res["Hedges' g"]
            eff = res["Effect size"]
            md  = res["mean diff"]
            row[label] = f"mean={md:+.4f}  {hg:.2f} ({eff})"

        df = pd.DataFrame([row], index=["Ensemble mean (3/4 agree)"])
        print(df.to_string())
   
'''
############
#############
#---on a map---
from scipy.stats import ttest_ind, mannwhitneyu
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

sig_maps_t   = {case: {} for case in cases}
sig_maps_mwu = {case: {} for case in cases}
effect_maps  = {case: {} for case in cases}

min_n_sd   = 3
min_n_nosd = 5

for case in cases:
    print(f"\n--- {case} ---")

    # --- Load ---
    if case == "ERA5L_lpjml":
        gpp_anom  = xr.open_dataarray(base_dir + "LPJmL_gpp_anom.nc")
        SWEI      = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
    else:
        SWEI      = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        gpp_anom  = xr.open_dataarray(base_dir + case + "_gpp_anom.nc")
        swe_month = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]

    gpp_anom = gpp_anom.sel(time=slice(start_y, end_y)).where(valid_land_case)
    SWEI     = SWEI.sel(time=slice(start_y, end_y)).where(valid_land_case)
    gpp_anom = gpp_anom / 10**12

    sd_mask_yr = (SWEI <= -1.3).rename({"time": "year"})
    sd_mask_yr = sd_mask_yr.sel(lat=sd_mask_yr.lat >= 25)
    sd_mask_yr = sd_mask_yr.transpose("year", "lat", "lon")

    swc_start_month  = (swe_month + 1).clip(max=9)
    spring_end_month = (swe_month + 3).clip(max=9)

    growing_ok = xr.apply_ufunc(
        lambda m, start: (m >= start) & (m <= 9),
        gpp_anom["time.month"], swc_start_month,
        input_core_dims=[[], []],
        vectorize=True, dask="parallelized", output_dtypes=[bool])

    spring_ok = xr.apply_ufunc(
        lambda m, s, e: (m >= s) & (m <= e),
        gpp_anom["time.month"], swc_start_month, spring_end_month,
        input_core_dims=[[], [], []],
        vectorize=True, dask="parallelized", output_dtypes=[bool])

    summer_ok = gpp_anom["time.month"].isin([6, 7, 8, 9])

    season_windows = {
        "Growing Season": growing_ok,
        "Spring":         spring_ok,
        "Summer":         summer_ok,
    }

    for season_name, s_ok in season_windows.items():
        print(f"  {season_name}...")

        gpp_s        = gpp_anom.where(s_ok, drop=False).where(valid_land_case)
        gpp_s_yearly = gpp_s.groupby("time.year").sum(
            dim="time", skipna=True, min_count=1)

        # --- Align spatial grids ---
        common_years = np.intersect1d(
            gpp_s_yearly.year.values, sd_mask_yr.year.values)
        gpp_yr_xr = gpp_s_yearly.sel(year=common_years)
        sd_yr_xr  = sd_mask_yr.sel(year=common_years)
        sd_yr_xr  = sd_yr_xr.sel(lat=gpp_yr_xr.lat, lon=gpp_yr_xr.lon)

        gpp_yr = gpp_yr_xr.values   # (n_years, lat, lon)
        sd_yr  = sd_yr_xr.values    # (n_years, lat, lon)

        n_lat, n_lon = gpp_yr.shape[1], gpp_yr.shape[2]
        
        # === Count pixels meeting SD threshold ===
        sd_count_per_pixel   = sd_yr.sum(axis=0)              # (lat, lon) — n SD years per pixel
        nosd_count_per_pixel = (~sd_yr).sum(axis=0)           # (lat, lon) — n noSD years per pixel
        valid_pixels         = ~np.isnan(gpp_yr[0, :, :])    # pixels with any data
    
        meets_sd_thresh   = (sd_count_per_pixel   >= min_n_sd)   & valid_pixels
        meets_nosd_thresh = (nosd_count_per_pixel >= min_n_nosd) & valid_pixels
        meets_both        = meets_sd_thresh & meets_nosd_thresh
    
        n_valid  = int(valid_pixels.sum())
        n_sd_ok  = int(meets_sd_thresh.sum())
        n_both   = int(meets_both.sum())
    
        print(f"    Valid land pixels:              {n_valid}")
        print(f"    Pixels with >={min_n_sd} SD years:      "
              f"{n_sd_ok} ({100*n_sd_ok/max(n_valid,1):.1f}%)")
        print(f"    Pixels meeting both thresholds: "
              f"{n_both} ({100*n_both/max(n_valid,1):.1f}%)")
        print(f"    SD years per pixel — "
              f"mean: {sd_count_per_pixel[valid_pixels].mean():.1f}, "
              f"min: {sd_count_per_pixel[valid_pixels].min()}, "
              f"max: {sd_count_per_pixel[valid_pixels].max()}")

        p_t_map   = np.full((n_lat, n_lon), np.nan)
        p_mwu_map = np.full((n_lat, n_lon), np.nan)
        d_map     = np.full((n_lat, n_lon), np.nan)

        for i in range(n_lat):
            for j in range(n_lon):
                gpp_ts = gpp_yr[:, i, j]
                sd_ts  = sd_yr[:, i, j]

                valid_t   = ~np.isnan(gpp_ts)
                sd_vals   = gpp_ts[sd_ts  & valid_t]
                nosd_vals = gpp_ts[~sd_ts & valid_t]

                if len(sd_vals) < min_n_sd or len(nosd_vals) < min_n_nosd:
                    continue

                _, p_t   = ttest_ind(sd_vals, nosd_vals, equal_var=False)
                _, p_mwu = mannwhitneyu(sd_vals, nosd_vals, alternative="two-sided")

                pooled_std = np.sqrt(
                    (np.std(sd_vals, ddof=1)**2 + np.std(nosd_vals, ddof=1)**2) / 2)
                d = (np.mean(sd_vals) - np.mean(nosd_vals)) / pooled_std if pooled_std > 0 else np.nan

                p_t_map[i, j]   = p_t
                p_mwu_map[i, j] = p_mwu
                d_map[i, j]     = d

        coords = {"lat": gpp_yr_xr.lat, "lon": gpp_yr_xr.lon}
        sig_maps_t[case][season_name]   = xr.DataArray(p_t_map,   dims=["lat", "lon"], coords=coords)
        sig_maps_mwu[case][season_name] = xr.DataArray(p_mwu_map, dims=["lat", "lon"], coords=coords)
        effect_maps[case][season_name]  = xr.DataArray(d_map,     dims=["lat", "lon"], coords=coords)

        n_sig_t   = int(np.sum(p_t_map   < 0.05))
        n_sig_mwu = int(np.sum(p_mwu_map < 0.05))
        n_total   = int(np.sum(~np.isnan(p_t_map)))
        print(f"    t-test  sig pixels: {n_sig_t}/{n_total} ({100*n_sig_t/max(n_total,1):.1f}%)")
        print(f"    MWU     sig pixels: {n_sig_mwu}/{n_total} ({100*n_sig_mwu/max(n_total,1):.1f}%)")


# === Plot: significance map + effect size map per case per season ===
for season_name in ["Growing Season", "Spring", "Summer"]:
    fig, axes = plt.subplots(
        len(cases), 2, figsize=(16, 4 * len(cases)),
        subplot_kw={"projection": ccrs.Robinson()})
    fig.suptitle(f"Per-pixel significance & effect size — {season_name}", fontsize=14)

    for row_i, case in enumerate(cases):
        p_map    = sig_maps_mwu[case][season_name]
        d_map_da = effect_maps[case][season_name]

        # --- Left: significance map ---
        ax = axes[row_i, 0]
        ax.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.LAND,  facecolor="lightgrey", zorder=1)

        sig_mask  = (p_map < 0.05)
        diff_case = diff_data[case]

        pos_sig = diff_case.where(sig_mask & (diff_case > 0))
        neg_sig = diff_case.where(sig_mask & (diff_case < 0))

        pos_sig.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=mcolors.ListedColormap(["#4393c3"]),
            add_colorbar=False, zorder=2)
        neg_sig.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=mcolors.ListedColormap(["#d6604d"]),
            add_colorbar=False, zorder=2)
        
        # --- SD density contour overlay (≥3 SD years) ---
        sd_count_map = sd_yr_xr.values.sum(axis=0).astype(float)
        sd_count_map[sd_count_map < 3] = np.nan  # mask below threshold
        
        sd_count_xr = xr.DataArray(
            sd_count_map,
            dims=["lat", "lon"],
            coords={"lat": gpp_yr_xr.lat, "lon": gpp_yr_xr.lon},
        )
        
        # Regrid to a regular 1-degree grid for smooth contouring
        lat_reg = np.arange(25, 90, 1.0)
        lon_reg = np.arange(-180, 180, 1.0)
        sd_reg  = sd_count_xr.interp(lat=lat_reg, lon=lon_reg, method="linear")
        
        LON, LAT = np.meshgrid(lon_reg, lat_reg)
        data_vals = sd_reg.values
        
        ax.contour(
            LON, LAT, data_vals,
            levels=[3, 5, 8, 11],          # adjust to your data range
            colors=["black"],
            linewidths=[0.6, 0.9, 1.2, 1.5],
            linestyles=["dashed"],
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        ax.contourf(
            LON, LAT, data_vals,
            levels=[3, 99],                 # shade areas with ≥3 SD years
            colors=["none"],
            hatches=["...."],
            transform=ccrs.PlateCarree(),
            zorder=4,
            alpha=0.0,
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=3)
        ax.add_feature(cfeature.LAKES, facecolor="white", linewidth=0.3, zorder=3)

        n_sig = int(sig_mask.values.sum())
        n_tot = int((~np.isnan(p_map.values)).sum())
        ax.text(0.03, 0.06,
                f"{n_sig}/{n_tot} sig\n({100*n_sig/max(n_tot,1):.1f}%)",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(f"{case} — MWU p<0.05 | blue=SD↑GPP  red=SD↓GPP", fontsize=10)

        # --- Right: Cohen's d map ---
        ax2 = axes[row_i, 1]
        ax2.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax2.add_feature(cfeature.LAND,  facecolor="lightgrey", zorder=1)

        mappable = d_map_da.plot.pcolormesh(
            ax=ax2, transform=ccrs.PlateCarree(),
            cmap="coolwarm_r", vmin=-1.0, vmax=1.0,
            add_colorbar=False, zorder=2)

        ax2.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)
        ax2.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=3)
        ax2.add_feature(cfeature.LAKES, facecolor="white", linewidth=0.3, zorder=3)
        ax2.set_title(f"{case} — Cohen's d  (±1 scale)", fontsize=10)

        plt.colorbar(mappable, ax=ax2, orientation="horizontal",
                     pad=0.02, fraction=0.04, label="Cohen's d")
        

    plt.tight_layout()
    plt.show()
    
    #########
'''
