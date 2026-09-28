import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import pandas as pd


# ================================================================
# config
# ================================================================
base_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
cases    = ["ERA5L_lpjml", "fluxcom", "nirv", "vodca2"]
start_y  = "1988"
end_y    = "2018"
sd_threshold = -1.3
gpp_units = "[TgC/grid cell]"
unit_qnty = 10**12

#plot fonts
font_title = 15
font_deg = 13
font_leg = 14

#options to plot 
gpp_to_plot = "all"     # "all" or "neg" or "Summer" or "Spring"
diff_to_plot = "3agree" # "mean" or "3agree" or "4 agree"

#colors of the datasets
case_colors = {
    "ERA5L_lpjml": "#4472C4",
    "fluxcom":     "#E4572E",
    "nirv":        "#2CA02C",
    "vodca2":      "#9467BD"
}

#display label for lpjml
display_labels = {"ERA5L_lpjml": "LPJmL","fluxcom": "FLUXCOM","nirv": "NIRv", "vodca2":"VODCA2"}

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

# ================================================================
# masks
# ================================================================
#greenland mask
greenland = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks\greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0) 

#land mask
ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"]
ocean_mask = ocean_mask.isel(time=0)
land_mask = (ocean_mask == 1) 

#threshold snow
threshold_mask_lpjml = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"]
threshold_mask_lpjml = (threshold_mask_lpjml == 1)
threshold_mask_era5l = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"]
threshold_mask_era5l = (threshold_mask_era5l == 1)

#cropland counts
croplands = xr.open_dataset(r"C:\Users\m.varela\Documents\Data_SnowDrought\masks\MODIS_2016_landcoverRegions_remapped.nc")["var02"] 
croplands = croplands.sel(lat=croplands.lat >= 25)
croplands = croplands.isel(time=0, lev=0, drop=True)
no_cropland_mask = croplands != 12


#combining masks to get the valid land area
valid_land_base = (land_mask & greenland & threshold_mask_lpjml & threshold_mask_era5l).copy(deep=True)
valid_land_crops = (land_mask & greenland & threshold_mask_lpjml & threshold_mask_era5l & no_cropland_mask).copy(deep=True)

# Number of grid cells classified as cropland (value == 12) within your existing valid_land_base mask
cropland_pixels_within_valid = int((valid_land_base & (croplands == 12)).sum().values)

# Total valid land pixels before the cropland exclusion
total_valid_pixels_before = int(valid_land_base.sum().values)

# Total valid pixels after excluding cropland
total_valid_pixels_after = int(valid_land_crops.sum().values)

print(f"Valid land pixels before cropland mask: {total_valid_pixels_before}")
print(f"Cropland pixels excluded: {cropland_pixels_within_valid}")
print(f"Valid land pixels after cropland mask: {total_valid_pixels_after}")
print("percentage excluded:", (cropland_pixels_within_valid/total_valid_pixels_before)*100)

#eco mask and ai mask 
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


# Percentage of each ecoregion affected by cropland exclusion
print("\nCropland exclusion by ecoregion:")
for eco_id in eco_ids:
    eco_mask = (eco_map == eco_id)
    valid_in_eco = int((valid_land_base & eco_mask).sum().values)
    cropland_in_eco = int((valid_land_base & eco_mask & (croplands == 12)).sum().values)
    if valid_in_eco > 0:
        pct_cropland = (cropland_in_eco / valid_in_eco) * 100
    else:
        pct_cropland = np.nan
    label = ecolabels.get(eco_id, f"Eco {eco_id}")
    print(f"{label}: {cropland_in_eco}/{valid_in_eco} pixels excluded ({pct_cropland:.1f}%)")
    
# Percentage of each aridity class affected by cropland exclusion
print("\nCropland exclusion by aridity class:")
for ai_id in ai_ids:
    ai_mask = (ai_map == ai_id)
    valid_in_ai = int((valid_land_base & ai_mask).sum().values)
    cropland_in_ai = int((valid_land_base & ai_mask & (croplands == 12)).sum().values)
    if valid_in_ai > 0:
        pct_cropland = (cropland_in_ai / valid_in_ai) * 100
    else:
        pct_cropland = np.nan
    label = AI_labels.get(ai_id, f"AI {ai_id}")
    print(f"{label}: {cropland_in_ai}/{valid_in_ai} pixels excluded ({pct_cropland:.1f}%)")

diff_data = {}

#===============================================
# per case swe/gpp analysis
#===============================================
for case in cases:

    # Load data ===============================================================
    if case == "ERA5L_lpjml":
        gpp_anom       = xr.open_dataarray(base_dir + "LPJmL_gpp_anom.nc")
        SWEI           = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month      = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        
    else:
        SWEI           = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        gpp_anom       = xr.open_dataarray(base_dir + case + "_gpp_anom.nc")
        swe_month      = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
    
    #peak-month exclusion
    swe_month_valid = swe_month.where(valid_land_base)
    peak_month_ok   = (swe_month_valid <= 9) | swe_month_valid.isnull()
    peak_month_ok   = peak_month_ok.reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
    valid_land_case = valid_land_base & peak_month_ok

    n_excluded = int((valid_land_base & ~peak_month_ok).sum())
    n_baseline = int(valid_land_base.sum())
    print(f"[{case}] excluding {n_excluded}/{n_baseline} pixels ({100*n_excluded/n_baseline:.2f}%) with non-physical peak SWE month (>9)")
    
    # Apply masks =============================================================
    gpp_anom = gpp_anom.sel(time=slice(start_y, end_y)).where(valid_land_case)
    SWEI     = SWEI.sel(time=slice(start_y, end_y)).where(valid_land_case)
    
    # making GPP units of TgC 
    gpp_anom = gpp_anom / unit_qnty

    # === Growing season mask =================================================
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
          

    # === SD mask: (year, lat, lon) ===========================================
    sd_mask_yr = (SWEI <= sd_threshold).rename({"time": "year"})  # (year, lat, lon)

    # === Per-pixel mean GPP per SD / noSD event ==============================
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
    
    '''
    # ===separate plots showing sum results for each dataset =============================================================
    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})
    
    # Check the actual range first
    print(float(diff.min()), float(diff.max()))
    
    # Then plot with symmetric colorbar centered on 0
    abs_max = float(np.nanpercentile(np.abs(diff.values), 95))
    
    mappable = diff.plot(ax=ax, cmap="coolwarm_r", vmin=-abs_max, vmax=abs_max,transform=ccrs.PlateCarree(), add_colorbar=False)
    
    ax.set_title(f"SD − noSD Mean Cumulative GPP Anomaly\n{case}", fontsize=font_title)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="none")
    
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.7)
    gl.xlabel_style = {"size": font_deg}
    gl.ylabel_style = {"size": font_deg}
    gl.bottom_labels = False
    gl.right_labels  = False
    
    cbar = plt.colorbar(mappable, orientation="horizontal", pad=0.02, fraction=0.05, aspect=70)
    cbar.set_label(f"∆GPP anomaly {gpp_units}", fontsize=font_leg)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.xaxis.set_ticks_position("bottom")
    
    plt.tight_layout()
    plt.show()
    '''      
#==============================================================================
# data analysis
#==============================================================================
#Stack into (case, lat, lon) 
diff_stack = xr.concat([diff_data[c] for c in cases], dim="case")  # (4, lat, lon)

# Metrics
diff_mean  = diff_stack.mean(dim="case", skipna=True)       # mean across datasets
n_negative = (diff_stack < 0).sum(dim="case")               # how many say SD worse
n_positive = (diff_stack > 0).sum(dim="case")               # how many say SD better

agreement  = n_negative - n_positive                         # -4 to +4

diff_agreed_3 = diff_mean.where((n_negative >= 3) | (n_positive >= 3))

diff_agreed_4 = diff_mean.where((n_negative == 4) | (n_positive == 4))

if gpp_to_plot == "all":
    gpp_title = "Growing season anomalies"
    sub = "$_{gs}$"

elif gpp_to_plot == "Summer":
    gpp_title = "Summer anomalies"
    sub = "$_{su}$"
    
elif gpp_to_plot == "Spring":
    gpp_title = "Spring anomalies"
    sub = "$_{sp}$"

if diff_to_plot == "mean":
    diff_input = diff_mean
    title      = "Mean ∆GPP"
    
elif diff_to_plot == "3agree":
    diff_input = diff_agreed_3
    title      = " ∆GPP"+ sub + " (≥3/4 datasets agree)"
    
elif diff_to_plot == "4agree":
    diff_input = diff_agreed_4
    title      = "Mean ∆GPP (4 datasets agree)"
    
    
#Plot Agreement map: shows where how many of the 4 datasets agree on sign (+ or -)
def agreement_map():
    
    colors    = ['#2166ac','#92c5de','#d1e5f0','#f7f7f7','#fddbc7','#f4a582','#d6604d','#b2182b']
    cmap_agree = mcolors.LinearSegmentedColormap.from_list("agree", colors, N=9)
    
    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})
    mappable2 = agreement.plot(ax=ax, cmap=cmap_agree, vmin=-4.5, vmax=4.5, transform=ccrs.PlateCarree(), add_colorbar=False)
    
    ax.set_title("Dataset Agreement: N datasets saying SD worse − SD better", fontsize=font_title)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="none")
    
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    gl.bottom_labels = False
    gl.right_labels  = False
    gl.xlabel_style = {"size": font_deg}
    gl.ylabel_style = {"size": font_deg}
    
    cbar = plt.colorbar(mappable2, orientation="horizontal", pad=0.02, fraction=0.05, aspect=70, ticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    cbar.set_label("Agreement score (−4 = all SD better, +4 = all SD worse)", fontsize=font_leg)
    cbar.ax.set_xticklabels(['-4', '-3', '-2', '-1', '0', '+1', '+2', '+3', '+4'])
    cbar.ax.tick_params(labelsize=font_leg) 
    
    plt.tight_layout()
    plt.show()


def mean_4datasets():

    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})
    mappable = diff_mean.plot(ax=ax, cmap="coolwarm_r", vmin=-0.125, vmax=0.125, transform=ccrs.PlateCarree(), add_colorbar=False)
    
    ax.set_title("Mean SD − noSD GPP Anomaly across 4 datasets", fontsize=font_title)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="none")
    
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    gl.bottom_labels = False
    gl.right_labels  = False
    gl.xlabel_style = {"size": font_deg}
    gl.ylabel_style = {"size": font_deg}
    
    cbar = plt.colorbar(mappable, orientation="horizontal", pad=0.02, fraction=0.05, aspect=70)
    cbar.set_label(f"Mean ∆GPP anomaly {gpp_units}", fontsize=font_leg)
    cbar.ax.tick_params(labelsize=font_leg) 
    
    plt.tight_layout()
    plt.show()

#Plot: Mean diff only where ALL 4 agree on sign OR 3/4 agree (depending on diff_input)
def mean_4datasets_filter():
    
    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})
    
    abs_max   = float(np.nanpercentile(np.abs(diff_input.values), 95))
    mappable3 = diff_input.plot(ax=ax, cmap="coolwarm_r", vmin=-abs_max, vmax=abs_max, transform=ccrs.PlateCarree(), add_colorbar=False)

    ax.coastlines()
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="none")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    
    gl.bottom_labels = False
    gl.right_labels  = False
    gl.xlabel_style = {"size": font_deg}
    gl.ylabel_style = {"size": font_deg}
    
    cbar = plt.colorbar(mappable3, orientation="horizontal", pad=0.02, fraction=0.05, aspect=70)
    cbar.set_label(title + f" {gpp_units}", fontsize=font_leg)
    
    per_case_neg = {}
    per_case_pos = {}
    
    for c in cases:
        v = diff_data[c].values
        nn = int(np.sum(v < 0))
        np_ = int(np.sum(v > 0))
        nt = nn + np_
        per_case_neg[c] = 100 * nn / nt if nt > 0 else np.nan
        per_case_pos[c] = 100 * np_ / nt if nt > 0 else np.nan
    
    vals    = diff_input.values
    n_neg   = int(np.sum(vals < 0))
    n_pos   = int(np.sum(vals > 0))
    n_total = n_neg + n_pos
    
    '''
    neg_pct_mean = 100 * n_neg / n_total
    pos_pct_mean = 100 * n_pos / n_total
    neg_vals = [per_case_neg[c] for c in cases]
    pos_vals = [per_case_pos[c] for c in cases]
    
    #box_text = (
    #    f" {neg_pct_mean:.1f}% neg  [{min(neg_vals):.1f}–{max(neg_vals):.1f}%]\n"
    #    f" {pos_pct_mean:.1f}% pos  [{min(pos_vals):.1f}–{max(pos_vals):.1f}%]"
    #)
    
    #ax.text(0.03, 0.06, box_text, transform=ax.transAxes, fontsize=13, verticalalignment="bottom",
    #    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85),
    #    fontfamily="monospace")
    '''
    
    cbar.ax.tick_params(labelsize=font_leg) 
    plt.tight_layout()
    plt.show()

    print(f"\n[{gpp_to_plot}] {100*n_neg/n_total:.1f}% neg / {100*n_pos/n_total:.1f}% pos  (n={n_total})")

#plots for perc and accumulation
def data_ranges():

    # === compute stats (run before both figures) =====================================
    per_case_neg = {}
    per_case_pos = {}
    per_case_cum = {}
    
    # agreement mask, True where diff_input is valid (≥3/4 agree)
    agree_mask = ~np.isnan(diff_input.values)

    for c in cases:
        v  = diff_data[c].values
        v_agreed = v[agree_mask]                        
        nn  = int(np.sum(v_agreed < 0))
        np_ = int(np.sum(v_agreed > 0))
        nt  = nn + np_
        per_case_neg[c] = 100 * nn  / nt if nt > 0 else np.nan
        per_case_pos[c] = 100 * np_ / nt if nt > 0 else np.nan
        per_case_cum[c] = float(np.nansum(v_agreed))
    
    vals         = diff_input.values
    n_neg        = int(np.sum(vals < 0))
    n_pos        = int(np.sum(vals > 0))
    n_total      = n_neg + n_pos
    neg_pct_mean = 100 * n_neg  / n_total
    pos_pct_mean = 100 * n_pos  / n_total
    cum_mean     = float(np.nansum(vals))
    
    neg_vals = [per_case_neg[c] for c in cases]
    pos_vals = [per_case_pos[c] for c in cases]
    cum_vals = [per_case_cum[c] for c in cases]
    
    row_labels = cases + ["mean"]
    neg_all    = neg_vals + [neg_pct_mean]
    pos_all    = pos_vals + [pos_pct_mean]
    cum_all    = cum_vals + [cum_mean]
   
    y_pos = np.arange(len(row_labels))
    bar_h = 0.3
    
    plot_labels = [display_labels.get(label, label) for label in row_labels]
    
    
    #Plot 4a: % neg / pos bars 
    
    fig_neg, ax_bar = plt.subplots(figsize=(6, 4))
    
    for i, (label, neg, pos) in enumerate(zip(row_labels, neg_all, pos_all)):
        color   = case_colors.get(label, "#888888")
        is_mean = (label == "mean")
        fc = color if not is_mean else "#888888"
        ec = color if not is_mean else "none"
        lw = 1.2   if not is_mean else 0
        
        ax_bar.barh(i, -neg, height=bar_h, color=fc, edgecolor=ec, linewidth=lw, alpha=0.85)
        ax_bar.barh(i,  pos, height=bar_h, color=fc, edgecolor=ec, linewidth=lw, alpha=0.4)
        ax_bar.text(-neg - 0.8, i, f"{neg:.1f}%", va="center", ha="right", fontsize=11)
        ax_bar.text( pos + 0.8, i, f"{pos:.1f}%", va="center", ha="left",  fontsize=11)
    
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.axhline(len(cases) - 0.5, color="gray", linewidth=0.8, linestyle="--")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(plot_labels, fontsize=7) #row_labels
    ax_bar.set_xlabel("% of ∆GPP"+sub+" grid cells \n (neg ←|→ pos)", fontsize=12)
    ax_bar.set_xlim(-105, 105)
    ax_bar.set_xticks([-100, -50, 0, 50, 100])
    ax_bar.set_xticklabels(["100", "50", "0", "50", "100"], fontsize=11)
    ax_bar.tick_params(axis="both", labelsize=11)
    ax_bar.axvspan(-105,   0, alpha=0.04, color="red")
    ax_bar.axvspan(   0, 105, alpha=0.04, color="blue")
    ax_bar.spines[["top", "right"]].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # === Plot 4b: cumulative GPP bars ================================
    fig_cum, ax_cum = plt.subplots(figsize=(6, 4))
    
    for i, (label, cum) in enumerate(zip(row_labels, cum_all)):
        color = case_colors.get(label, "#888888")
        is_mean = (label == "mean")
        fc  = color if not is_mean else "#888888"
        
        ax_cum.barh(i, cum, height=bar_h, alpha=0.85 if cum < 0 else 0.4, color=fc, edgecolor=fc, linewidth=1.2 if not is_mean else 0)
        
        if gpp_to_plot == "all" and label == "nirv":
            ax_cum.text( 5, i, f"{cum:.2f}", va="center", ha="left",fontsize=font_deg)
        else:
           ax_cum.text(cum * 1.02, i, f"{cum:.2f}", va="center", ha="left" if cum >= 0 else "right", fontsize=11)
    
    ax_cum.axvline(0, color="black", linewidth=0.8)
    ax_cum.axhline(len(cases) - 0.5, color="gray", linewidth=0.8, linestyle="--")
    ax_cum.set_yticks(y_pos)
    ax_cum.set_yticklabels(plot_labels, fontsize=7)
    ax_cum.set_xlabel("Cumul. ∆GPP"+sub+ "[TgC] \n(≥3/4 agreed grid cells)", fontsize=12)
    ax_cum.tick_params(axis="both", labelsize=11)
    ax_cum.spines[["top", "right"]].set_visible(False)
    
    if gpp_to_plot == "Summer":
        ax_cum.set_xlim(-350,0)
        
    xlim = ax_cum.get_xlim()
    ax_cum.axvspan(xlim[0], 0, alpha=0.04, color="red")
    ax_cum.axvspan(0, xlim[1], alpha=0.04, color="blue")
    #plt.tight_layout()
    plt.show()

data_ranges()

# ECOREGION and AI BARPLOTS
def eco_ai_barplots():
    
    eco_means, eco_medians, eco_iqr_low, eco_iqr_high, eco_names = [], [], [], [], []
    ai_means, ai_medians, ai_iqr_low, ai_iqr_high, ai_names = [], [], [], [], []

    for eco_id in eco_ids:
        emask = (eco_map == eco_id)
        vals  = diff_input.where(emask).values #here the input is the mean of differences
        vals  = vals[~np.isnan(vals)]
    
        if len(vals) > 0:
            
            q25, q75 = np.percentile(vals, [25, 75])
            eco_means.append(np.mean(vals))
            eco_medians.append(np.median(vals))
            eco_iqr_low.append(q25)   # absolute Q25
            eco_iqr_high.append(q75)  # absolute Q75
            
            # percentage of pixels above and below 0
            pct_neg = 100 * np.sum(vals < 0) / len(vals)
            pct_pos = 100 * np.sum(vals > 0) / len(vals)
            
            name = ecolabels.get(eco_id, f"Eco {eco_id}")
            eco_names.append(f"{name}\n({pct_neg:.0f}% neg / {pct_pos:.0f}% pos)")
    
    eco_means   = np.array(eco_means)
    eco_medians = np.array(eco_medians)
    eco_iqr_low = np.array(eco_iqr_low)
    eco_iqr_high= np.array(eco_iqr_high)
    order       = np.argsort(eco_means)
    colors_bar  = ['#d6604d' if v < 0 else '#4393c3' for v in eco_means[order]] #red if sd has negative impact, blue otherwise
    
    df_eco = pd.DataFrame({
        "ecoregion": eco_names,
        "mean": eco_means,
        "median": eco_medians,
        "q25": eco_iqr_low,
        "q75": eco_iqr_high,
    })
    
    #sort by mean
    df_eco = df_eco.sort_values("mean").reset_index(drop=True)
    
    for ai_id in ai_ids:
        amask = (ai_map == ai_id)
        vals  = diff_input.where(amask).values
        vals  = vals[~np.isnan(vals)]
        
        if len(vals) > 0:
            
            q25, q75 = np.percentile(vals, [25, 75])
            ai_means.append(np.mean(vals))
            ai_medians.append(np.median(vals))
            ai_iqr_low.append(q25)   # absolute Q25
            ai_iqr_high.append(q75)  # absolute Q75
          
            # percentage of pixels above and below 0
            pct_neg = 100 * np.sum(vals < 0) / len(vals)
            pct_pos = 100 * np.sum(vals > 0) / len(vals)
            
            name = AI_labels.get(ai_id, f"AI {ai_id}")
            ai_names.append(f"{name}\n({pct_neg:.0f}% neg / {pct_pos:.0f}% pos)")
             
    ai_means    = np.array(ai_means)
    ai_medians  = np.array(ai_medians)
    ai_iqr_low  = np.array(ai_iqr_low)
    ai_iqr_high = np.array(ai_iqr_high)
    order_ai    = np.argsort(ai_means)
    colors_bar_ai = ['#d6604d' if v < 0 else '#4393c3' for v in ai_means[order_ai]]
    
    
    #Plot ecoregion summary barplot GPP ======
    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.barh([eco_names[i] for i in order], eco_means[order],color=colors_bar, alpha=0.75, edgecolor='k', linewidth=0.5)
    
    # Draw IQR whiskers manually, centered on median
    for i, idx in enumerate(order):
        ax.plot([eco_iqr_low[idx], eco_iqr_high[idx]], [i, i], color='black', linewidth=0.8, solid_capstyle='round')
        # end caps
        ax.plot([eco_iqr_low[idx], eco_iqr_low[idx]], [i-0.05, i+0.05], color='black', linewidth=0.8)
        ax.plot([eco_iqr_high[idx], eco_iqr_high[idx]], [i-0.05, i+0.05],color='black', linewidth=0.8)
    
    # Median line
    ax.scatter(eco_medians[order], range(len(order)), color='black', marker='|', s=100, zorder=5, label='Median')
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel(f"∆GPP"+sub +f" {gpp_units}", fontsize=font_leg)
    #ax.set_title(title + "\n" +gpp_title+ "\n(bar=mean, whiskers=IQR, line=median)", fontsize=12)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=font_deg)
    ax.tick_params(axis='y', labelsize=font_deg)
    plt.tight_layout()
    plt.show()

    #Plot aridity index summary barplot GPP =======
    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.barh([ai_names[i] for i in order_ai], ai_means[order_ai], color=colors_bar_ai, alpha=0.75, edgecolor='k', linewidth=0.5)

    # Draw IQR whiskers manually, centered on median
    for i, idx in enumerate(order_ai):
        ax.plot([ai_iqr_low[idx], ai_iqr_high[idx]], [i, i],color='black', linewidth=0.8, solid_capstyle='round')
        # end caps
        ax.plot([ai_iqr_low[idx], ai_iqr_low[idx]], [i-0.05, i+0.05],color='black', linewidth=0.8)
        ax.plot([ai_iqr_high[idx], ai_iqr_high[idx]], [i-0.05, i+0.05],color='black', linewidth=0.8)

    # Median line
    ax.scatter(ai_medians[order_ai], range(len(order_ai)),color='black', marker='|', s=100, zorder=5, label='Median')
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel(f"∆GPP"+sub +f" {gpp_units}", fontsize=font_leg)
    #ax.set_title(title + "\n" +gpp_title+ "\n(bar=mean, whiskers=IQR, line=median)", fontsize=12)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    ax.tick_params(axis='x', labelsize=font_deg)
    ax.tick_params(axis='y', labelsize=font_deg)
    plt.show()
    
    

eco_ai_barplots()

'''
def dataset_summ_plots():
 
    #Multi-dataset grouped bar
    results = {case: {"medians": [], "means": [], "iqr_low": [], "iqr_high": [], "names": []} for case in cases}
    
    eco_order_fixed   = [13, 8, 4, 10, 5, 6, 11]
    eco_ids_ordered   = [eid for eid in eco_order_fixed if eid in eco_ids]
    order_eco         = [list(eco_ids).index(eid) for eid in eco_ids_ordered]
    eco_names_ordered = [ecolabels.get(eid, f"Eco {eid}") for eid in eco_ids_ordered]
    
    for eco_id in eco_ids_ordered:
        emask = (eco_map == eco_id)
        for case in cases:
            vals = diff_data[case].where(emask).values
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                q25, med, q75 = np.percentile(vals, [25, 50, 75])
                results[case]["medians"].append(med)
                results[case]["means"].append(np.mean(vals))
                results[case]["iqr_low"].append(med - q25)
                results[case]["iqr_high"].append(q75 - med)
                results[case]["names"].append(ecolabels.get(eco_id, f"Eco {eco_id}"))
            else:
                results[case]["medians"].append(np.nan)
                results[case]["means"].append(np.nan)
                results[case]["iqr_low"].append(np.nan)
                results[case]["iqr_high"].append(np.nan)
                results[case]["names"].append(ecolabels.get(eco_id, f"Eco {eco_id}"))
    
    n_cases = len(cases)
    n_ecos  = len(eco_ids_ordered)
    bar_h   = 0.18
    offsets = np.linspace(-(n_cases-1)/2, (n_cases-1)/2, n_cases) * bar_h
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for ci, case in enumerate(cases):
        means    = np.array(results[case]["means"])
        medians  = np.array(results[case]["medians"])
        iqr_low  = np.array(results[case]["iqr_low"])
        iqr_high = np.array(results[case]["iqr_high"])
        y_pos    = np.arange(n_ecos) + offsets[ci]
    
        ax.barh(y_pos, means, height=bar_h,
                #xerr=[iqr_low, iqr_high],
                color=case_colors[case], alpha=0.75,
                edgecolor='k', linewidth=0.4,
                capsize=2, error_kw={'linewidth': 0.8},
                label=case)
        ax.scatter(medians, y_pos,
                   color='black', marker='|',
                   s=80, linewidths=1.2, zorder=5)
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(np.arange(n_ecos))
    ax.set_yticklabels(eco_names_ordered, fontsize=10)
    ax.set_xlabel(f"SD − noSD GPP {gpp_units}", fontsize=10)
    ax.set_title("SD − noSD per Ecoregion per Dataset\n(bar=mean, line=median)\n" + gpp_title, fontsize=11)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()



    # === Plot 2: dataset summary barplots ==================================
    results_ai = {case: {"medians": [], "means": [], "iqr_low": [], "iqr_high": [], "names": []} for case in cases}
    
    ai_order_fixed   = [1, 2, 3, 4, 5]
    ai_ids_ordered   = [aid for aid in ai_order_fixed if aid in ai_ids]
    ai_names_ordered = [AI_labels.get(aid, f"AI {aid}") for aid in ai_ids_ordered]
    
    for ai_id in ai_ids_ordered:
        amask = (ai_map == ai_id)
        for case in cases:
            vals = diff_data[case].where(amask).values
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                q25, med, q75 = np.percentile(vals, [25, 50, 75])
                results_ai[case]["medians"].append(med)
                results_ai[case]["means"].append(np.mean(vals))
                results_ai[case]["iqr_low"].append(med - q25)
                results_ai[case]["iqr_high"].append(q75 - med)
                results_ai[case]["names"].append(AI_labels.get(ai_id, f"AI {ai_id}"))
            else:
                results_ai[case]["medians"].append(np.nan)
                results_ai[case]["means"].append(np.nan)
                results_ai[case]["iqr_low"].append(np.nan)
                results_ai[case]["iqr_high"].append(np.nan)
                results_ai[case]["names"].append(AI_labels.get(ai_id, f"AI {ai_id}"))
    
    n_cases = len(cases)
    n_ai    = len(ai_ids_ordered)
    bar_h   = 0.18
    offsets = np.linspace(-(n_cases-1)/2, (n_cases-1)/2, n_cases) * bar_h
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for ci, case in enumerate(cases):
        means    = np.array(results_ai[case]["means"])
        medians  = np.array(results_ai[case]["medians"])
        iqr_low  = np.array(results_ai[case]["iqr_low"])
        iqr_high = np.array(results_ai[case]["iqr_high"])
        y_pos    = np.arange(n_ai) + offsets[ci]
    
        ax.barh(y_pos, means, height=bar_h,
                #xerr=[iqr_low, iqr_high],
                color=case_colors[case], alpha=0.75,
                edgecolor='k', linewidth=0.4,
                capsize=2, error_kw={'linewidth': 0.8},
                label=case)
        ax.scatter(medians, y_pos,
                   color='black', marker='|',
                   s=80, linewidths=1.2, zorder=5)
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(np.arange(n_ai))
    ax.set_yticklabels(ai_names_ordered, fontsize=10)
    ax.set_xlabel(f"SD − noSD GPP {gpp_units}", fontsize=10)
    ax.set_title("SD − noSD per Aridity Index per Dataset\n(bar=mean, line=median)\n" + gpp_title, fontsize=11)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
    
'''