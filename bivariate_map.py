import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# === Paths =============================================================================================
base_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
cases    = ["ERA5L_lpjml", "fluxcom", "nirv", "vodca2"]
start_y  = "1988"
end_y    = "2018"

ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"]
ocean_mask = ocean_mask.isel(time=0)
land_mask = (ocean_mask == 1) 

# === options to plot =====================
diff_to_plot = "3agree" # "mean" or "3agree" or "4 agree"
gpp_to_plot = "Summer" #Spring"#"Summer"

# === Labels / colors =======
#colors of the datasets
case_colors = {
    "ERA5L_lpjml": "#4472C4",
    "fluxcom":     "#E4572E",
    "nirv":        "#2CA02C",
    "vodca2":      "#9467BD"
}

font_leg = 14
font_deg = 13

diff_data = {}

############################
#
#
### bivariate map ####
#
#
#############################
# Run agreement filtering independently per season, then combine signs.
# Requires: diff_data_spring and diff_data_summer dicts already populated
# (see below — recompute both in one pass)

diff_data_spring = {}
diff_data_summer = {}

for case in cases:

    if case == "ERA5L_lpjml":
        gpp_anom  = xr.open_dataarray(base_dir + "LPJmL_gpp_anom.nc")
        SWEI      = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        
    else:
        gpp_anom  = xr.open_dataarray(base_dir + case + "_gpp_anom.nc")
        SWEI      = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        
    threshold_mask_lpjml = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"] == 1
    threshold_mask_era5l = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"] == 1

    threshold_mask = threshold_mask_lpjml & threshold_mask_era5l

    gpp_anom = gpp_anom.where(threshold_mask).sel(time=slice(start_y, end_y))
    SWEI     = SWEI.where(threshold_mask).sel(time=slice(start_y, end_y))

    # === Spring (dynamic window) ===
    start_m  = (swe_month + 1).clip(max=9)
    end_m    = (swe_month + 3).clip(max=9)
    month_ok = xr.apply_ufunc(
        lambda m, s, e: (m >= s) & (m <= e),
        gpp_anom["time.month"], start_m, end_m,
        vectorize=True, dask="parallelized", output_dtypes=[bool],
    )
    gpp_spring = gpp_anom.where(month_ok, drop=False)

    # === Summer (fixed JJA) ===
    gpp_summer = gpp_anom.where(gpp_anom["time.month"].isin([6, 7, 8,9]), drop=True)

    # === Compute diff (with division-by-zero guard) ===
    def compute_diff(gpp_input, sd_mask_yr):
        gpp_yearly = gpp_input.groupby("time.year").sum(dim="time", skipna=True, min_count=1)
        sd_count   = sd_mask_yr.sum("year")
        nosd_count = (~sd_mask_yr).sum("year")
        sd_mean    = (gpp_yearly.where(sd_mask_yr).sum("year", skipna=True)
                      .where(sd_count > 0) / sd_count.where(sd_count > 0))
        nosd_mean  = (gpp_yearly.where(~sd_mask_yr).sum("year", skipna=True)
                      .where(nosd_count > 0) / nosd_count.where(nosd_count > 0))
        return sd_mean - nosd_mean

    sd_mask_yr = (SWEI <= -1.3).rename({"time": "year"})

    diff_data_spring[case] = compute_diff(gpp_spring, sd_mask_yr)
    diff_data_summer[case] = compute_diff(gpp_summer, sd_mask_yr)


# === Stack and apply 3/4 agreement independently per season ===
spring_stack = xr.concat([diff_data_spring[c] for c in cases], dim="case")
summer_stack = xr.concat([diff_data_summer[c] for c in cases], dim="case")

# Mean across datasets
spring_mean = spring_stack.median(dim="case", skipna=True).where(threshold_mask)
summer_mean = summer_stack.median(dim="case", skipna=True).where(threshold_mask)

# 3/4 agreement filter — applied independently
n_neg_sp = (spring_stack < 0).sum("case")
n_pos_sp = (spring_stack > 0).sum("case")
n_neg_su = (summer_stack < 0).sum("case")
n_pos_su = (summer_stack > 0).sum("case")

spring_agreed = spring_mean#.where((n_neg_sp >= 3) | (n_pos_sp >= 3))
summer_agreed = summer_mean#.where((n_neg_su >= 3) | (n_pos_su >= 3))

# === Bivariate classification ===
valid_mask = threshold_mask & spring_agreed.notnull() & summer_agreed.notnull()

spring_sign = xr.where(spring_agreed > 0, 1, -1)
summer_sign = xr.where(summer_agreed > 0, 1, -1)

bivar = xr.full_like(spring_agreed, np.nan)
bivar = xr.where(valid_mask & (spring_sign == -1) & (summer_sign == -1), 0, bivar)  # --
bivar = xr.where(valid_mask & (spring_sign == -1) & (summer_sign ==  1), 1, bivar)  # -+
bivar = xr.where(valid_mask & (spring_sign ==  1) & (summer_sign == -1), 2, bivar)  # +-
bivar = xr.where(valid_mask & (spring_sign ==  1) & (summer_sign ==  1), 3, bivar)  # ++

# === Colormap ===
colors = [
    "#d73027",  # -- (Spring -, Summer -)
    "#e7d4e8",  # -+ (Spring -, Summer +)
    "#7b3294",  # +- (Spring +, Summer -)
    "#0466c8",  # ++ (Spring +, Summer +)
]
cmap = mcolors.ListedColormap(colors)


# === Plot ===
fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={"projection": ccrs.Robinson()})

bivar.plot(ax=ax, cmap=cmap, vmin=-0.5, vmax=3.5,transform=ccrs.PlateCarree(), add_colorbar=False)

#ax.set_title("Bivariate Map: Spring vs Summer SD impact on GPP", fontsize=13) #\n(≥3/4 agreement applied independently per season)
ax.coastlines()
ax.add_feature(cfeature.LAND, edgecolor="black", facecolor= "#EFEFDB")
ax.add_feature(cfeature.LAKES,edgecolor="black", facecolor="none")
ax.coastlines(zorder=2)

gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
gl.bottom_labels = False
gl.right_labels  = False
gl.xlabel_style = {"size": font_deg}
gl.ylabel_style = {"size": font_deg}

vals        = bivar.values
total_valid = int(np.sum(~np.isnan(vals)))

labels_count = [
    f"Neg. Spring, Neg. Summer (--): {100*np.sum(vals==0)/total_valid:.1f}%",
    f"Neg. Spring, Pos. Summer (-+): {100*np.sum(vals==1)/total_valid:.1f}%",
    f"Pos. Spring, Neg. Summer (+-): {100*np.sum(vals==2)/total_valid:.1f}%",
    f"Pos. Spring, Pos. Summer (++): {100*np.sum(vals==3)/total_valid:.1f}%",
]

patches = [mpatches.Patch(color=colors[i], label=labels_count[i]) for i in range(4)]
ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.08),ncol=2, fontsize=font_leg, frameon=False)

plt.tight_layout()
plt.show()

