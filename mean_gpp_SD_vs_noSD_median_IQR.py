import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import sys
sys.path.append(r'C:\Users\m.varela\Documents\Data_SnowDrought')
from myfunctions import anomalies

#=== Paths ============================================================================================
base_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
case     = "fluxcom"
version  = "dynamic"
out_dir  = base_dir + case + "_" + version + "/"

color_sd   = '#4472C4'
color_nosd = '#1E6B1E'
start_y    = "1988"
end_y      = "2018"

cases = ["ERA5L_lpjml","nirv","vodca2","fluxcom"]

case_labels = {
    "ERA5L_lpjml": "LPJmL",
    "nirv":        "NIRv",
    "vodca2":      "VODCA2",
    "fluxcom":     "FLUXCOM"
}


case_colors = {
    'ERA5L_lpjml': '#3B6BC1',
    'nirv':        '#2CA02C',
    'vodca2':      '#9467BD',
    'fluxcom':     '#E3562D',
}

#plot fonts
font_title = 15
font_leg = 14

gpp_units = "[TgC/grid cell]"
gpp_units_sum = "[TgC]"

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
months       = np.arange(1, 10)   # Jan=1 ... Oct=10
USE_MEDIAN   = True # switch to False for mean+std

# Season definitions (indices into the 10-element monthly arrays, Jan=idx0 ... Oct=idx9)
# used only for the seasonal summary tables below.
SEASON_SLICES = {
    'Spring (Mar-May)':  slice(2, 5),   # months 3,4,5
    'Summer (Jun-Aug)':  slice(5, 8),   # months 6,7,8
    'Net (Mar-Oct)':     slice(2, 10),  # months 3-10
}
 
all_cases_nh      = {}
all_cases_nh_sum  = {}  #summed TgC version of all_cases_nh
all_cases_eco     = {}  #per-ecoregion, mean [TgC/grid cell]
all_cases_ai      = {}  #per-aridity-class, mean [TgC/grid cell]

# =============================================================================
# ECOREGION & ARIDITY INDEX DEFINITIONS
# =============================================================================
ecolabels = {
    4:  "Temp. Broadleaf & Mixed Forests",
    5:  "Temp. Conifer Forests",
    6:  "Boreal Forests/Taiga",
    8:  "Temp. Grasslands, Sav. & Shrublands",
    10: "Montane Grasslands & Shrublands",
    11: "Tundra",
    13: "Deserts and Xeric Shrublands",
}
SKIP_ECOS = {1, 2, 3, 7, 9, 12, 14}

AI_labels = {
    1: "Hyper Arid",
    2: "Arid",
    3: "Semi-Arid",
    4: "Dry Sub-Humid",
    5: "Humid",
}

# =============================================================================
# LOAD STATIC MAPS 
# =============================================================================
eco_map = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/Terrestrial Ecoregions of the World/data/commondata/data0/wwf_ecoregions_0.5deg_remap.nc")["ecoregion"]
ai_map = xr.open_dataset(r"C:/Users/m.varela/Documents/Data_SnowDrought/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/aridity_index_class.nc")["aridity_index"]
 
eco_ids = np.unique(eco_map.values)
eco_ids = eco_ids[~np.isnan(eco_ids)].astype(int)
eco_ids = eco_ids[eco_ids != 0]
eco_ids = np.array([e for e in eco_ids if e not in SKIP_ECOS])
print("Ecoregion IDs found:", eco_ids)
eco_valid_mask = xr.DataArray(np.isin(eco_map.values, eco_ids), dims=eco_map.dims, coords=eco_map.coords)
 
ai_ids = np.unique(ai_map.values)
ai_ids = ai_ids[~np.isnan(ai_ids)].astype(int)
ai_ids = ai_ids[ai_ids != 0]
print("Aridity Index IDs found:", ai_ids)

#=== Core function ============================================================

def get_monthly_distribution(gpp, sd_mask, is_sd, mode="mean", spatial_mask=None):
    """
    mode="mean"  -> each yearly value is the per-grid-cell average GPP anomaly
                    (sum of valid pixels / n_valid) -> units TgC/grid cell
    mode="total" -> each yearly value is the SUM of GPP anomaly across all valid
                    pixels in the mask (no division) -> units TgC
    """
    monthly_median, monthly_q25, monthly_q75 = [], [], []
    monthly_mean, monthly_std = [], []
 
    for m in months:
        gpp_m  = gpp.sel(time=gpp.time.dt.month == m)
        yearly = []
 
        for t in gpp_m.time:
            yr = int(t.dt.year)
            if yr not in sd_mask.time.values:
                continue
 
            sd_yr     = sd_mask.sel(time=yr)
            gpp_yr    = gpp_m.sel(time=t)
            cond_mask = sd_yr if is_sd else ~sd_yr
 
            if spatial_mask is not None:
                cond_mask = cond_mask & spatial_mask
 
            valid_mask = cond_mask
            n_valid = int(valid_mask.sum())
            if n_valid > 0:
                total = float(gpp_yr.where(valid_mask).sum())
                yearly.append(total if mode == "total" else total / n_valid)
 
        yearly = np.array(yearly)
        yearly = yearly[~np.isnan(yearly)]
 
        if len(yearly) > 0:
            monthly_median.append(np.median(yearly))
            monthly_q25.append(np.percentile(yearly, 25))
            monthly_q75.append(np.percentile(yearly, 75))
            monthly_mean.append(np.mean(yearly))
            monthly_std.append(np.std(yearly))
        else:
            for lst in [monthly_median, monthly_q25, monthly_q75, monthly_mean, monthly_std]:
                lst.append(np.nan)
 
    return {
        'median': np.array(monthly_median),
        'q25':    np.array(monthly_q25),
        'q75':    np.array(monthly_q75),
        'mean':   np.array(monthly_mean),
        'std':    np.array(monthly_std),
    }


# =========================================================================
# COMPUTE PER CASE
# =========================================================================

for case in cases:
    
    #=== Data =================================================================
    if case == "ERA5L_lpjml":
        gpp_anom = xr.open_dataarray(base_dir + "LPJmL_gpp_anom.nc")
        SWEI = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        threshold_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"] == 1
        
         
    else:
        SWEI = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        gpp_anom = xr.open_dataarray(base_dir + case + "_gpp_anom.nc")
        swe_month = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        threshold_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"] == 1

    
    #=== Apply masks ==========================================================
    gpp_anom  = gpp_anom.where(threshold_mask & eco_valid_mask).sel(time=slice(start_y, end_y))
    SWEI = SWEI.where(threshold_mask & eco_valid_mask).sel(time=slice(start_y, end_y))
    eco_map_masked = eco_map.where(threshold_mask & eco_valid_mask)
    ai_map_masked  = ai_map.where(threshold_mask & eco_valid_mask)
    
    # === making GPP units of TgC ======
    gpp_anom = gpp_anom / 10**12
    
    
    #=== SD mask and growing season ===========================================
    sd_mask  = SWEI <= -1.3
    swc_start_month = (swe_month + 1).clip(max=9)
    month_da = gpp_anom["time.month"]
    
    month_ok = xr.apply_ufunc(
        lambda m, start: (m >= start) & (m <= 9),
        month_da, swc_start_month,
        input_core_dims=[[], []],
        vectorize=True, dask="parallelized", output_dtypes=[bool],
    )
    gpp_season = gpp_anom.where(month_ok, drop=False)
    
    # === Compute NH ========================================================
    print(f"Computing NH ({case})")
    nh = {
        'sd_all':   get_monthly_distribution(gpp_season, sd_mask, is_sd=True),
        'nosd_all': get_monthly_distribution(gpp_season, sd_mask, is_sd=False),
    }
 
    all_cases_nh[case] = nh
    
    #total (TgC) NH distribution
    nh_sum = {
        'sd_all':   get_monthly_distribution(gpp_season, sd_mask, is_sd=True,  mode="total"),
        'nosd_all': get_monthly_distribution(gpp_season, sd_mask, is_sd=False, mode="total"),
    }
    all_cases_nh_sum[case] = nh_sum
    
    #per ecoregion mean (TgC/grid cell)
    print(f"  Computing per-ecoregion ({case})...")
    eco_results = {}
    for eco_id in eco_ids:
        emask = (eco_map_masked == eco_id)
        eco_results[eco_id] = {
            'sd_all':   get_monthly_distribution(gpp_season, sd_mask, is_sd=True,  spatial_mask=emask),
            'nosd_all': get_monthly_distribution(gpp_season, sd_mask, is_sd=False, spatial_mask=emask),
        }
    all_cases_eco[case] = eco_results
 
    # per aridity class (mean, TgC/grid cell)
    print(f"  Computing per-aridity-class ({case})...")
    ai_results = {}
    for ai_id in ai_ids:
        amask = (ai_map_masked == ai_id)
        ai_results[ai_id] = {
            'sd_all':   get_monthly_distribution(gpp_season, sd_mask, is_sd=True,  spatial_mask=amask),
            'nosd_all': get_monthly_distribution(gpp_season, sd_mask, is_sd=False, spatial_mask=amask),
        }
    all_cases_ai[case] = ai_results


# =========================================================================
# PLOT: SD panel and no-SD panel, one line per dataset, shared y-axis
# =========================================================================
keys = ['sd_all', 'nosd_all']
 
# shared y-limits across both plots
all_lowers, all_uppers = [], []
for key in keys:
    for case in cases:
        data = all_cases_nh[case][key]
        all_lowers.append(np.nanmin(data['q25']))
        all_uppers.append(np.nanmax(data['q75']))
y_min = min(all_lowers)
y_max = max(all_uppers)
y_pad = 0.05 * (y_max - y_min)
y_lims = (y_min - y_pad, y_max + y_pad)
 
 
def plot_dataset_comparison(key, title_suffix):
    fig, ax = plt.subplots(figsize=(7, 5))
    for case in cases:
        data   = all_cases_nh[case][key]
        color  = case_colors[case]
        label  = case_labels[case]
        center = data['median']
        lower  = data['q25']
        upper  = data['q75']
        ax.plot(months, center, color=color, linewidth=1.8, marker='o', markersize=4, label=label)
        ax.fill_between(months, lower, upper, color=color, alpha=0.15)
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':', alpha=0.4)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, fontsize=14)
    ax.set_xlabel('Month', fontsize=font_leg)
    ax.set_ylabel(f'Median GPP Anomaly {gpp_units}', fontsize=font_leg)
    ax.set_ylim(y_lims)
    ax.tick_params(axis='y', labelsize=font_leg)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

 
plot_dataset_comparison('sd_all', 'Snow Drought')
plot_dataset_comparison('nosd_all', 'No Snow Drought')

# =========================================================================
#  PLOT: NH, total (TgC)
# =========================================================================

#SD
fig_sd_total, ax_sd_total = plt.subplots(figsize=(7, 5))
for case in cases:
    data   = all_cases_nh_sum[case]['sd_all']
    color  = case_colors[case]
    label  = case_labels[case]
    center = data['median']
    lower  = data['q25']
    upper  = data['q75']
    ax_sd_total.plot(months, center, color=color, linewidth=1.8, marker='o', markersize=4, label=label)
    ax_sd_total.fill_between(months, lower, upper, color=color, alpha=0.15)
ax_sd_total.axhline(0, color='black', linewidth=0.7, linestyle=':', alpha=0.4)
ax_sd_total.set_title('Snow Drought', fontsize=12)
ax_sd_total.set_xticks(months)
ax_sd_total.set_xticklabels(month_labels, fontsize=11)
ax_sd_total.set_xlabel('Month', fontsize=12)
ax_sd_total.set_ylabel(f'Median GPP Anomaly {gpp_units_sum}', fontsize=12)
ax_sd_total.set_ylim(-45,70)
ax_sd_total.yaxis.grid(True, linestyle='--', alpha=0.4)
ax_sd_total.set_axisbelow(True)
ax_sd_total.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.show()

'''
#NO SD
fig_nosd_total, ax_nosd_total = plt.subplots(figsize=(7, 5))
for case in cases:
    data   = all_cases_nh_sum[case]['nosd_all']
    color  = case_colors[case]
    label  = case_labels[case]
    center = data['median']
    lower  = data['q25']
    upper  = data['q75']
    ax_nosd_total.plot(months, center, color=color, linewidth=1.8, marker='o', markersize=4, label=label)
    ax_nosd_total.fill_between(months, lower, upper, color=color, alpha=0.15)
ax_nosd_total.axhline(0, color='black', linewidth=0.7, linestyle=':', alpha=0.4)
ax_nosd_total.set_title('No Snow Drought', fontsize=12)
ax_nosd_total.set_xticks(months)
ax_nosd_total.set_xticklabels(month_labels, fontsize=11)
ax_nosd_total.set_xlabel('Month', fontsize=12)
ax_nosd_total.set_ylabel(f'Median GPP Anomaly {gpp_units_sum}', fontsize=12)
ax_nosd_total.set_ylim(-200,200)
ax_nosd_total.yaxis.grid(True, linestyle='--', alpha=0.4)
ax_nosd_total.set_axisbelow(True)
ax_nosd_total.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.show()
'''

# =========================================================================
#PLOT: per-ecoregion grid, mean [TgC/grid cell] Snow Drought
# =========================================================================
ncols_eco = 3
nrows_eco = int(np.ceil(len(eco_ids) / ncols_eco))
fig_eco_sd, axes_eco_sd = plt.subplots(nrows_eco, ncols_eco, figsize=(ncols_eco * 5, nrows_eco * 3.8), sharey=True)
axes_eco_sd_flat = axes_eco_sd.flatten()
 
for i, eco_id in enumerate(eco_ids):
    ax = axes_eco_sd_flat[i]
    for case in cases:
        data = all_cases_eco[case][eco_id]['sd_all']
        ax.plot(months, data['median'], color=case_colors[case], linewidth=1.5, marker='o', markersize=3, label=case_labels[case])
        ax.fill_between(months, data['q25'], data['q75'], color=case_colors[case], alpha=0.15)
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':', alpha=0.4)
    ax.set_title(ecolabels.get(eco_id, str(eco_id)), fontsize=font_leg)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, fontsize=font_leg)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    if i % ncols_eco == 0:
        ax.set_ylabel(f'Median GPP Anom. {gpp_units}', fontsize=font_leg)
 
for j in range(len(eco_ids), len(axes_eco_sd_flat)):
    axes_eco_sd_flat[j].set_visible(False)
 
handles, labels = axes_eco_sd_flat[0].get_legend_handles_labels()
fig_eco_sd.legend(handles, labels, fontsize=font_leg, loc='lower center', ncol=len(cases), bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()

# =========================================================================
# PLOT: per-aridity-class grid, mean [TgC/grid cell] Snow Drought
# =========================================================================
ncols_ai = 3
nrows_ai = int(np.ceil(len(ai_ids) / ncols_ai))
fig_ai_sd, axes_ai_sd = plt.subplots(nrows_ai, ncols_ai, figsize=(ncols_ai * 5, nrows_ai * 3.8), sharey=True)
axes_ai_sd_flat = axes_ai_sd.flatten()
 
for i, ai_id in enumerate(ai_ids):
    ax = axes_ai_sd_flat[i]
    for case in cases:
        data = all_cases_ai[case][ai_id]['sd_all']
        ax.plot(months, data['median'], color=case_colors[case], linewidth=1.5, marker='o', markersize=3, label=case_labels[case])
        ax.fill_between(months, data['q25'], data['q75'], color=case_colors[case], alpha=0.15)
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':', alpha=0.4)
    ax.set_title(AI_labels.get(ai_id, str(ai_id)), fontsize=font_leg)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, fontsize=font_leg)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='y', labelsize=font_leg)
    ax.set_axisbelow(True)
    if i % ncols_ai == 0:
        ax.set_ylabel(f'Median GPP Anom. {gpp_units}', fontsize=font_leg)
 
for j in range(len(ai_ids), len(axes_ai_sd_flat)):
    axes_ai_sd_flat[j].set_visible(False)
 
handles, labels = axes_ai_sd_flat[0].get_legend_handles_labels()
fig_ai_sd.legend(handles, labels, fontsize=font_leg, loc='lower center', ncol=len(cases), bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()


#############
def print_monthly_table(results_dict, label_dict, condition_key='sd_all', title=""):
    """
    Prints median [Q25–Q75] GPP anomaly per grid cell for each month and dataset.
    """
    print(f"\n{'='*100}")
    print(f"  {title}  |  condition: {condition_key}")
    print(f"  Values = monthly median [Q25–Q75] GPP anomaly [TgC/grid cell]")
    print(f"{'='*100}")

    # header: months
    header = f"  {'Dataset':<14}" + "".join(f"  {m:>18}" for m in month_labels)
    print(header)
    print(f"  {'-'*94}")

    for case in results_dict:
        data = results_dict[case].get(condition_key)
        if data is None:
            continue
        row = f"  {label_dict[case]:<14}"
        for i in range(len(months)):
            med = data['median'][i]
            q25 = data['q25'][i]
            q75 = data['q75'][i]
            row += f"  {med:>+6.4f} [{q25:+.4f}–{q75:+.4f}]"
        print(row)

    # ensemble summary
    print(f"  {'-'*94}")
    ens_med_row  = f"  {'Ens. median':<14}"
    ens_mean_row = f"  {'Ens. mean':<14}"
    for i in range(len(months)):
        vals = [results_dict[c][condition_key]['median'][i]
                for c in results_dict
                if results_dict[c].get(condition_key) is not None]
        vals = [v for v in vals if not np.isnan(v)]
        ens_med_row  += f"  {np.nanmedian(vals):>+6.4f}{'':>18}"
        ens_mean_row += f"  {np.nanmean(vals):>+6.4f}{'':>18}"
    print(ens_med_row)
    print(ens_mean_row)
    print()


# =============================================================================
# CALL IT
# =============================================================================
print("\n" + "="*100)
print("  MONTHLY STATS: Northern Hemisphere")
print("="*100)
for cond_key, cond_name in [('sd_all', 'Snow Drought'), ('nosd_all', 'No Snow Drought')]:
    print_monthly_table(
        {case: all_cases_nh[case] for case in cases},
        case_labels, condition_key=cond_key,
        title=f"NH — {cond_name}",
    )
    
######
def seasonal_stats(data):
    """
    Given a distribution dict (monthly arrays of length 10, Jan–Oct),
    compute the SUM of monthly medians for Spring, Summer, and Net season.
    Also sums Q25/Q75 to give a rough seasonal IQR range.

    Returns: { season_name: { 'sum_median', 'sum_q25', 'sum_q75' } }
    """
    out = {}
    for sname, sl in SEASON_SLICES.items():
        out[sname] = {
            'sum_median': np.nansum(data['median'][sl]),
            'sum_q25':    np.nansum(data['q25'][sl]),
            'sum_q75':    np.nansum(data['q75'][sl]),
        }
    return out


def print_stats_table(results_dict, label_dict, condition_key='sd_all', title=""):
    """
    For each dataset: prints summed seasonal GPP anomaly (Spring / Summer / Net)
    as the sum of monthly median values within the season window.
    Then prints ensemble median and mean across datasets.

    The summed monthly medians represent the cumulative GPP anomaly [TgC]
    over that season, averaged across SD (or no-SD) years.
    """
    print(f"\n{'='*80}")
    print(f"  {title}  |  condition: {condition_key}")
    print(f"  Values = sum of monthly medians [TgC] (Spring=Mar–May, Summer=Jun–Aug, Net=Mar–Oct)")
    print(f"{'='*80}")
    header = f"  {'Dataset':<14}" + "".join(f"  {s:>22}" for s in SEASON_SLICES)
    print(header)
    print(f"  {'-'*76}")

    all_vals = {s: [] for s in SEASON_SLICES}

    for case in results_dict:
        data = results_dict[case].get(condition_key)
        if data is None:
            continue
        ss = seasonal_stats(data)
        row = f"  {label_dict[case]:<14}"
        for sname in SEASON_SLICES:
            med = ss[sname]['sum_median']
            q25 = ss[sname]['sum_q25']
            q75 = ss[sname]['sum_q75']
            row += f"  {med:>7.3f} [{q25:.3f}–{q75:.3f}]"
            all_vals[sname].append(med)
        print(row)

    print(f"  {'-'*76}")
    ens_med_row  = f"  {'Ens. median':<14}"
    ens_mean_row = f"  {'Ens. mean':<14}"
    for sname in SEASON_SLICES:
        vals = [v for v in all_vals[sname] if not np.isnan(v)]
        ens_med_row  += f"  {np.nanmedian(vals):>7.3f}{'':>14}"
        ens_mean_row += f"  {np.nanmean(vals):>7.3f}{'':>14}"
    print(ens_med_row)
    print(ens_mean_row)
    print()

print("\n" + "="*80)
print("  STATS: Northern Hemisphere (all pixels)")
print("="*80)
for cond_key, cond_name in [('sd_all', 'Snow Drought'), ('nosd_all', 'No Snow Drought')]:
    print_stats_table(
        {case: all_cases_nh[case] for case in cases},
        case_labels, condition_key=cond_key,
        title=f"NH — {cond_name}",
    )