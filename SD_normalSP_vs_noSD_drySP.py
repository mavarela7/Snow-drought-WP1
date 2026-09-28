import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =============================================================================
# CONFIG
# =============================================================================
base_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
cases    = ["ERA5L_lpjml", "fluxcom", "nirv", "vodca2"]
start_y  = "1988"
end_y    = "2018"

gpp_units = "[TgC/grid cell]"
font_leg = 14
font_deg = 13

# =============================================================================
# MASKS
# =============================================================================
greenland = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/greenland_mask.nc")["greenland_mask"]
greenland = (greenland.sel(lat=greenland.lat >= 25) == 0)

ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"].isel(time=0)
land_mask  = (ocean_mask == 1)

threshold_mask_lpjml = (xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"] == 1)
threshold_mask_era5l = (xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"] == 1)

valid_land_base = (land_mask & greenland & threshold_mask_lpjml & threshold_mask_era5l).copy(deep=True)

# =============================================================================
# REGION LABELS
# =============================================================================
ecolabels = {
    4: "Broadleaf & Mixed Forests", 5: "Temp. Conifer Forests",
    6: "Boreal Forests/Taiga",            8: "Temp. Grasslands",
    10: "Montane Grasslands",             11: "Tundra",
    13: "Deserts and Xeric",
}

SKIP_ECOS = {1, 2, 3, 7, 9, 12, 14}

ai_labels = {1: "Hyper Arid", 2: "Arid", 3: "Semi-Arid",
    4: "Dry Sub-Humid", 5: "Humid",}

eco_order_merged = [13, 8, 4, 10, 5, (6, 11)]
eco_merge_labels = {(6, 11): "Boreal/Tundra"}

ai_order_merged = [(1, 2), 3, 4, 5]
ai_merge_labels = {(1, 2): "Hyper Arid\n& Arid"}

eco_map = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/Terrestrial Ecoregions of the World/data/commondata/data0/wwf_ecoregions_0.5deg_remap.nc")["ecoregion"]
ai_map = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/aridity_index_class.nc")["aridity_index"]

eco_map = eco_map.sel(lat=eco_map.lat >= 25).reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
ai_map  = ai_map.sel(lat=ai_map.lat >= 25).reindex_like(valid_land_base, method="nearest", tolerance=1e-6)

def get_group_mask(group_map, gid):
    if isinstance(gid, tuple):
        return sum(group_map == g for g in gid).astype(bool)
    return (group_map == gid)

# =============================================================================
# MAIN LOOP, compute per-pixel mean GPP anomaly for Group A and B
# =============================================================================
maps_A = {}   # SD & normal spring
maps_B = {}   # noSD & dry spring

for case in cases:
    print(f"\n=== {case} ===")
    out_dir = base_dir + case + "_dynamic/"

    # --- load ---
    if case == "ERA5L_lpjml":
        gpp_anom  = xr.open_dataarray(base_dir + "LPJmL_gpp_anom.nc")
        SWEI      = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
    else:
        gpp_anom  = xr.open_dataarray(base_dir + case + "_gpp_anom.nc")
        SWEI      = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]

    # --- peak-month exclusion ---
    swe_month_valid = swe_month.where(valid_land_base)
    peak_month_ok   = (swe_month_valid <= 9) | swe_month_valid.isnull()
    peak_month_ok   = peak_month_ok.reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
    valid_land_case = valid_land_base & peak_month_ok

    n_excluded = int((valid_land_base & ~peak_month_ok).sum())
    n_baseline = int(valid_land_base.sum())
    print(f"  excluding {n_excluded}/{n_baseline} pixels ({100*n_excluded/n_baseline:.2f}%)")

    gpp_anom = gpp_anom.sel(time=slice(start_y, end_y)).where(valid_land_case).load() / 10**12
    SWEI     = SWEI.sel(time=slice(start_y, end_y)).where(valid_land_case).load()
    sd_mask_yr = (SWEI <= -1.3).rename({"time": "year"}).load()

    # --- dynamic spring window (peak SWE month + 1 to + 3, max month 9) ---
    start_m = (swe_month + 1).clip(max=9)
    end_m   = (swe_month + 3).clip(max=9)

    spring_ok = xr.apply_ufunc(
        lambda m, s, e: (m >= s) & (m <= e),
        gpp_anom["time.month"], start_m, end_m,
        input_core_dims=[[], [], []], vectorize=True,
        dask="parallelized", output_dtypes=[bool],
    )
    gpp_spring = gpp_anom.where(spring_ok, drop=False).where(valid_land_case).load()

    # --- load and align spring SPI masks ---
    spi_spring_neg = (xr.open_dataset(out_dir + "mask_spi_spring_-.nc")["spi_gamma_03"].transpose("time", "lat", "lon").rename({"time": "year"}))
    spi_spring_neg = spi_spring_neg.reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
    spi_spring_zero = (xr.open_dataset(out_dir + "mask_spi_spring_0.nc")["spi_gamma_03"].transpose("time", "lat", "lon").rename({"time": "year"}))
    spi_spring_zero = spi_spring_zero.reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
    
    # --- condition masks ---
    cond_A = sd_mask_yr.astype(bool)    & spi_spring_zero.astype(bool)   # SD & normal spring
    cond_B = (~sd_mask_yr.astype(bool)) & spi_spring_neg.astype(bool)    # noSD & dry spring

    # --- compute per-pixel mean GPP anomaly for each group ---
    vm        = valid_land_case.values
    gpp_np    = gpp_spring.values
    yr_arr    = gpp_spring.time.dt.year.values
    all_years = np.unique(yr_arr)
    nlat      = gpp_spring.sizes["lat"]
    nlon      = gpp_spring.sizes["lon"]

    for group_label, cond_mask, storage in [("A (SD & normal spring)", cond_A, maps_A),("B (noSD & dry spring)",  cond_B, maps_B),]:
        
        print(f"  Computing Group {group_label} …")
        yearly_maps = []

        for yr in all_years:
            if yr not in cond_mask.year.values:
                continue

            cond_yr = cond_mask.sel(year=yr).values
            active  = cond_yr & vm

            if not active.any():
                continue

            t_idx      = (yr_arr == yr)
            gpp_window = gpp_np[t_idx]

            masked  = np.where(active[np.newaxis, :, :], gpp_window, np.nan)
            gpp_sum = np.nansum(masked, axis=0)
            gpp_sum[np.all(np.isnan(masked), axis=0)] = np.nan
            yearly_maps.append(gpp_sum)

        if yearly_maps:
            mean_map = np.nanmean(np.stack(yearly_maps, axis=0), axis=0)
        else:
            mean_map = np.full((nlat, nlon), np.nan)

        storage[case] = xr.DataArray(mean_map, dims=["lat", "lon"],coords={"lat": gpp_spring.lat, "lon": gpp_spring.lon},)

# =============================================================================
# AGREEMENT FILTER (≥3/4 datasets agree on sign)
# =============================================================================
def agree(case_dict):
    stack = xr.concat([case_dict[c] for c in cases], dim="case")
    mean  = stack.mean(dim="case", skipna=True)
    n_neg = (stack < 0).sum(dim="case")
    n_pos = (stack > 0).sum(dim="case")
    return mean.where((n_neg >= 3) | (n_pos >= 3))

agreed_A = agree(maps_A)
agreed_B = agree(maps_B)

# =============================================================================
# BARPLOTS 
# =============================================================================
GROUP_COLORS = {"A": "#2166ac", "B": "#d94801"}
GROUP_LABELS = {"A": "SD & normal spring", "B": "noSD & dry spring"}

for group_name, group_ids, merge_lbl, base_lbl, gmap in [
    ("Ecoregion", eco_order_merged, eco_merge_labels, ecolabels, eco_map),
    ("Aridity Index Class",   ai_order_merged,  ai_merge_labels,  ai_labels,  ai_map),]:
    
    fig, ax = plt.subplots(figsize=(9, 7))

    bar_width = 0.32
    y = np.arange(len(group_ids))

    # --- collect stats for both groups ---
    meansA, mediansA, iqr_lowA, iqr_highA = [], [], [], []
    meansB, mediansB, iqr_lowB, iqr_highB = [], [], [], []

    for gid in group_ids:
        mask = get_group_mask(gmap, gid)

        vA = agreed_A.where(mask).values.ravel() 
        vA = vA[~np.isnan(vA)]
        vB = agreed_B.where(mask).values.ravel()
        vB = vB[~np.isnan(vB)]

        if len(vA) > 0:
            meansA.append(np.mean(vA))   
            mediansA.append(np.median(vA))
            iqr_lowA.append(np.percentile(vA, 25))
            iqr_highA.append(np.percentile(vA, 75))
        else:
            meansA.append(np.nan)
            mediansA.append(np.nan)
            iqr_lowA.append(np.nan)
            iqr_highA.append(np.nan)

        if len(vB) > 0:
            meansB.append(np.mean(vB))
            mediansB.append(np.median(vB))
            iqr_lowB.append(np.percentile(vB, 25))
            iqr_highB.append(np.percentile(vB, 75))
        else:
            meansB.append(np.nan)
            mediansB.append(np.nan)
            iqr_lowB.append(np.nan)
            iqr_highB.append(np.nan)

    meansA    = np.array(meansA)
    mediansA  = np.array(mediansA)
    iqr_lowA  = np.array(iqr_lowA)
    iqr_highA = np.array(iqr_highA)
    meansB    = np.array(meansB)
    mediansB  = np.array(mediansB)
    iqr_lowB  = np.array(iqr_lowB)
    iqr_highB = np.array(iqr_highB)

    cap_w = 0.05

    # --- draw vertical bars ---
    for offset, means, medians, iqr_low, iqr_high, group_key in [
        (-bar_width / 2, meansA, mediansA, iqr_lowA, iqr_highA, "A"),
        (+bar_width / 2, meansB, mediansB, iqr_lowB, iqr_highB, "B"),
    ]:
        xpos = y + offset
        ax.bar(xpos, means, bar_width,
               color=GROUP_COLORS[group_key], alpha=0.82,
               edgecolor="k", linewidth=0.5, label=GROUP_LABELS[group_key])

        for i in range(len(group_ids)):
            if not np.isnan(iqr_low[i]):
                ax.plot([xpos[i], xpos[i]], [iqr_low[i], iqr_high[i]], color="black", lw=0.8)
                ax.plot([xpos[i]-cap_w, xpos[i]+cap_w], [iqr_low[i],  iqr_low[i]],  "k-", lw=0.8)
                ax.plot([xpos[i]-cap_w, xpos[i]+cap_w], [iqr_high[i], iqr_high[i]], "k-", lw=0.8)
            ax.scatter([xpos[i]], [medians[i]], color="black", s=60, marker="_", lw=1.5, zorder=5)

    # --- x-tick labels with pixel counts ---
    xlabels = []
    for gid in group_ids:
        mask = get_group_mask(gmap, gid)
        nA   = int((~np.isnan(agreed_A.where(mask).values)).sum())
        nB   = int((~np.isnan(agreed_B.where(mask).values)).sum())
        name = merge_lbl.get(gid, base_lbl.get(gid, str(gid)))
        xlabels.append(f"{name}\n{nA} | {nB}")

    ax.axhline(0, color="black", lw=1.0, linestyle="--")
    ax.set_xticks(y)
    ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=font_leg)
    ax.set_ylabel(f"Mean spring GPP anomaly {gpp_units}", fontsize=font_leg)
    ax.set_title(group_name, fontsize=font_leg)
    ax.legend(fontsize=font_leg)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=font_leg)
    ax.annotate("n: SD & normal spring | noSD & dry spring",
                xy=(0.5, 0.03), xycoords="axes fraction",
                ha="center", fontsize=9, color="grey")

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAP 1 — difference (A − B): which condition is more productive
# Positive (blue): SD & normal spring more productive
# Negative (red):  noSD & dry spring more productive
# =============================================================================
diff_stack  = xr.concat([maps_A[c] - maps_B[c] for c in cases], dim="case")
diff_mean   = diff_stack.mean(dim="case", skipna=True)
n_neg       = (diff_stack < 0).sum(dim="case")
n_pos       = (diff_stack > 0).sum(dim="case")
diff_agreed = diff_mean.where((n_neg >= 3) | (n_pos >= 3))

fig_map, ax_map = plt.subplots(figsize=(14, 5), subplot_kw={"projection": ccrs.Robinson()})
ax_map.add_feature(cfeature.LAND, facecolor="#EFEFDB", zorder=0)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4)
ax_map.add_feature(cfeature.BORDERS, linewidth=0.3)
ax_map.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())

gl = ax_map.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
gl.bottom_labels = False
gl.right_labels  = False
gl.xlabel_style = {"size": font_deg}
gl.ylabel_style = {"size": font_deg}

vmax = float(np.nanpercentile(np.abs(diff_agreed.values[~np.isnan(diff_agreed.values)]), 95))
im = diff_agreed.plot(
    ax=ax_map, transform=ccrs.PlateCarree(),
    cmap="RdBu", vmin=-vmax, vmax=vmax,
    add_colorbar=False,
)
cbar = fig_map.colorbar(
    im, ax=ax_map,
    orientation="horizontal",
    fraction=0.05, pad=0.03, shrink=0.8, aspect=70,
    extend="both",
)
cbar.ax.tick_params(labelsize=font_deg)

cbar.set_label(f"(SD & normal spring) − (noSD & dry spring) GPP anomaly {gpp_units}", fontsize=font_leg)

# stats box
vals       = diff_agreed.values[~np.isnan(diff_agreed.values)]
n_total    = len(vals)
pct_pos    = 100 * np.sum(vals > 0) / n_total
pct_neg    = 100 * np.sum(vals < 0) / n_total
total_gpp  = float(np.nansum(vals))

box_text = (
    f"Positive (blue): {pct_pos:.1f}%\n"
    f"Negative (red):  {pct_neg:.1f}%"
)
ax_map.text(
    0.02, 0.05, box_text,
    transform=ax_map.transAxes,
    fontsize=font_deg, va="bottom", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85),
)

plt.tight_layout()
plt.show()

# =============================================================================
# MAPS 2 & 3: Group A and Group B separately, one figure each
# =============================================================================
all_vals = np.concatenate([
    agreed_A.values[~np.isnan(agreed_A.values)],
    agreed_B.values[~np.isnan(agreed_B.values)],
])
vmax = float(np.nanpercentile(np.abs(all_vals), 95))

for da, group_label in zip(
    [agreed_A, agreed_B],
    ["SD & normal spring", "noSD & dry spring"],
):
    fig_map, ax_map = plt.subplots(figsize=(14, 5), subplot_kw={"projection": ccrs.Robinson()})
    ax_map.add_feature(cfeature.LAND, facecolor="#EFEFDB", zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax_map.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())

    gl = ax_map.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    gl.bottom_labels = False
    gl.right_labels  = False
    gl.xlabel_style = {"size": font_deg}
    gl.ylabel_style = {"size": font_deg}

    im = da.plot(
        ax=ax_map, transform=ccrs.PlateCarree(),
        cmap="RdBu", vmin=-vmax, vmax=vmax,
        add_colorbar=False,
    )
    cbar = fig_map.colorbar(
        im, ax=ax_map,
        orientation="horizontal",
        fraction=0.05, pad=0.03, shrink=0.8, aspect=70,
        extend="both",
    )
    cbar.set_label(f"Mean spring GPP anomaly {gpp_units}", fontsize=font_leg)
    cbar.ax.tick_params(labelsize=font_deg)

    # stats + label box
    valid_vals = da.values[~np.isnan(da.values)]
    n_total    = len(valid_vals)
    pct_pos    = 100 * np.sum(valid_vals > 0) / n_total
    pct_neg    = 100 * np.sum(valid_vals < 0) / n_total
    total_gpp  = float(np.nansum(valid_vals))

    box_text = (
        f"{group_label}\n"
        f"Positive: {pct_pos:.1f}%\n"
        f"Negative: {pct_neg:.1f}%\n"
        f"Sum: {total_gpp:.3f} TgC"
    )
    ax_map.text(
        0.03, 0.05, box_text,
        transform=ax_map.transAxes,
        fontsize=font_deg, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85),
    )

    plt.tight_layout()
    plt.show()