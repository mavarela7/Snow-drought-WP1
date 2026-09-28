import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


# === Paths ===================================================================
base_dir = "C:/Users/m.varela/Documents/Data_SnowDrought/Analysis_final/"
cases    = ["ERA5L_lpjml", "fluxcom", "nirv", "vodca2"]
start_y  = "1988"
end_y    = "2018"

categories   = ['dry', 'normal', 'wet']
cat_colors   = {'dry': '#e74c3c', 'normal': '#95a5a6', 'wet': '#3498db'}
cat_labels   = {'dry': 'Dry', 'normal': 'Normal', 'wet': 'Wet'}
months       = np.arange(1, 11)
var_names    = ['Spring SWC', 'Summer SWC','Spring Precip', 'Summer Precip','Summer Precip with normal spring']

# === Window definitions ======================================================
spring_start, spring_end       = 4, 6
summer_start, summer_end       = 6, 8
summer_swc_start, summer_swc_end = 6, 9
swc_offset_before, swc_offset_after = 0, 0

#=== fonts ====================================================================
font_leg = 14
font_title = 15

gpp_units = "[TgC/grid cell]"

#=== chosing the months at the selected indices ===============================
summer_swc_idx   = [i for i, m in enumerate(months) if summer_swc_start <= m <= summer_swc_end]
spring_idx = [i for i, m in enumerate(months) if spring_start <= m <= spring_end]
summer_idx = [i for i, m in enumerate(months) if summer_start <= m <= summer_end]

# === Merged group definitions ================================================
eco_order_merged  = [13, 8, 4, 10, 5, (6, 11)] # (6,11) = Boreal+Tundra merged
eco_merge_labels  = {(6, 11): "Boreal/Tundra"}

ai_order_merged   = [(1, 2), 3, 4, 5]# (1,2) = HyperArid+Arid merged
ai_merge_labels   = {(1, 2): "Hyper Arid\n& Arid"}


def get_group_mask(group_map, gid):
    if isinstance(gid, tuple):
        return sum(group_map == g for g in gid).astype(bool)
    return (group_map == gid)

def get_group_label(gid, label_dict, fallback_dict):
    if gid in label_dict:
        return label_dict[gid]
    return fallback_dict.get(gid, str(gid))

# SWC window is per pixel so we define a function to get it per case
def get_swc_indices(swe_month_map):
    
    m_peak_swe  = int(np.nanmedian(swe_month_map.values))
    swc_start_month = (m_peak_swe + 1)
    swc_start_month = int(np.clip(swc_start_month, 1, 9))
    
    idx_center = swc_start_month - 1  # convert to 0-based
    idxs = [np.clip(idx_center - i, 0, 9) for i in range(swc_offset_before, 0, -1)]+ [ np.clip(idx_center + i, 0, 9) for i in range(0, swc_offset_after + 1)]
    
    return list(dict.fromkeys(idxs))


# === masks ===================================================================
#greenland mask
greenland = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\masks\greenland_mask.nc")["greenland_mask"]
greenland = greenland.sel(lat=greenland.lat >= 25)
greenland = (greenland == 0) 

#land mask
ocean_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/ocean_mask.nc")["ocean_mask"]
ocean_mask = ocean_mask.isel(time=0)
land_mask = (ocean_mask == 1) 

threshold_mask_lpjml = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"] == 1
threshold_mask_era5l = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"] == 1

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
    4: "Temp. Broadleaf & Mixed Forests",
    5: "Temp. Conifer Forests",
    6: "Boreal Forests/Taiga",
    8: "Temp. Grasslands",
    10: "Montane Grasslands",
    11: "Tundra",
    13: "Deserts and Xeric"
}

#not relevant ecolabels
SKIP_ECOS = {1,2,3,7,9,12,14}

# Aridity Index classes 
ai_labels = {
    1: "Hyper Arid",    # AI < 0.03
    2: "Arid",          # 0.03 – 0.2
    3: "Semi-Arid",     # 0.2  – 0.5
    4: "Dry Sub-Humid", # 0.5  – 0.65
    5: "Humid",         # > 0.65
}

# === eco mask and ai mask ====================================================
eco_map  = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/Terrestrial Ecoregions of the World/data/commondata/data0/wwf_ecoregions_0.5deg_remap.nc")["ecoregion"]
##eco_ids  = np.unique(eco_map.values)
##eco_ids  = eco_ids[~np.isnan(eco_ids)].astype(int)
##eco_ids  = eco_ids[eco_ids != 0]
##eco_ids  = np.array([e for e in eco_ids if e not in SKIP_ECOS])

ai_map  = xr.open_dataset("C:/Users\m.varela\Documents\Data_SnowDrought\Global-AI_ET0_annual_v3\Global-AI_ET0_v3_annual/aridity_index_class.nc")["aridity_index"]
##ai_ids = np.unique(ai_map.values)
##ai_ids = ai_ids[~np.isnan(ai_ids)].astype(int)
##ai_ids = ai_ids[ai_ids != 0]

eco_map = eco_map.sel(lat=eco_map.lat >= 25).reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
ai_map  = ai_map.sel(lat=ai_map.lat >= 25).reindex_like(valid_land_base, method="nearest", tolerance=1e-6)


#=== storing values ===========================================================
diff_data = {}
diff_maps_cat = {v: {cat: {} for cat in categories} for v in var_names}
##diff_maps_cat["Summer Precip with normal spring"] = {cat: {} for cat in categories}


# === function to calculate at pixel level, how does GPP differ between SD and noSD years
def compute_diff_map(gpp, sd_mask_yr, category_mask, valid_mask, month_indices):
    
    gpp_years    = np.unique(gpp.time.dt.year.values) #number of years in gpp file
    mask_years   = category_mask.year.values ##number of years in cat file
    common_years = np.intersect1d(gpp_years, mask_years) #making sure they share same years

    sd_maps, nosd_maps = [], []
    vm = valid_mask.values #making the valid_mask an array
    
    for yr in common_years:
        
        #for each year, we look:
        sd_yr  = sd_mask_yr.sel(year=yr).values #in what pixels there was a sd in year yr
        cat_yr = category_mask.sel(year=yr).values #in what pixels there was a given condition in year yr
        
        sd_cond   = sd_yr  & cat_yr & vm
        nosd_cond = (~sd_yr) & cat_yr & vm

        gpp_yr    = gpp.sel(time=gpp.time.dt.year == yr) #look at gpp values in year yr
        yr_months = gpp_yr.time.dt.month.values
        sel_idx   = [i for i, m in enumerate(yr_months) if m in [months[j] for j in month_indices]] #selecting the months given in month indices
        gpp_window = gpp_yr.isel(time=sel_idx).values #contains only the gpp values in the selected months

        def spatial_sum(cond):
            masked = np.where(cond[np.newaxis, :, :], gpp_window, np.nan) 
            result = np.nansum(masked, axis=0)
            # set back to NaN where all months were NaN
            all_nan = np.all(np.isnan(masked), axis=0)
            result[all_nan] = np.nan
            return result

        if sd_cond.any(): #if there is at least one pixel
            sd_maps.append(spatial_sum(sd_cond)) #gpp values only where sd_cond is true
        if nosd_cond.any():
            nosd_maps.append(spatial_sum(nosd_cond)) #gpp values only where nosd_cond is true

    if not sd_maps or not nosd_maps:
        empty = np.full(gpp.isel(time=0).shape, np.nan)
        return xr.DataArray(empty, dims=['lat', 'lon'], coords={'lat': gpp.lat, 'lon': gpp.lon})

    sd_mean = np.nanmean(np.stack(sd_maps,   axis=0), axis=0)
    nosd_mean = np.nanmean(np.stack(nosd_maps, axis=0), axis=0)
    
    return xr.DataArray(sd_mean - nosd_mean, dims=['lat','lon'],coords={'lat': gpp.lat, 'lon': gpp.lon})

#=== main loop ================================================================
for case in cases:
    out_dir = base_dir + case + "_dynamic/"

    # --- load ---
    if case == "ERA5L_lpjml":
        gpp_anom       = xr.open_dataarray(base_dir + "LPJmL_gpp_anom.nc")
        SWEI           = xr.open_dataset(base_dir + "LPJmL_SWEI_final_at_peak.nc")["SWEI_3month"]
        swe_month      = xr.open_dataset(base_dir + "/Peak_files/LPJmL_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        threshold_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/masks/threshold_mask.nc")["SWE"] == 1
        swc_var = "rootzone_swc"
    else:
        SWEI           = xr.open_dataset(base_dir + "ERA5L_SWEI_final_at_peak.nc")["SWEI_3month"]
        gpp_anom       = xr.open_dataarray(base_dir + case + "_gpp_anom.nc")
        swe_month      = xr.open_dataset(base_dir + "/Peak_files/ERA5L_maxSWEmonth_Mode_1988-2018.nc")["mode_peak_SWE_month"]
        threshold_mask = xr.open_dataset("C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_files/ERA5L_snow_threshold_mask.nc")["sd"] == 1
        swc_var = "swc_vol"
        
    swe_month_valid = swe_month.where(valid_land_base)
    peak_month_ok   = (swe_month_valid <= 9) | swe_month_valid.isnull()
    peak_month_ok   = peak_month_ok.reindex_like(valid_land_base, method="nearest", tolerance=1e-6)
    valid_land_case = valid_land_base & peak_month_ok
    
    n_excluded = int((valid_land_base & ~peak_month_ok).sum())
    n_baseline = int(valid_land_base.sum())
    print(f"[{case}] excluding {n_excluded}/{n_baseline} pixels "
          f"({100*n_excluded/n_baseline:.2f}%) with non-physical peak SWE month (>9)")

        
    gpp_anom = gpp_anom.sel(time=slice(start_y, end_y)).where(valid_land_case).load()
    SWEI     = SWEI.sel(time=slice(start_y, end_y)).where(valid_land_case).load()

    
    gpp_anom = gpp_anom / 10**12  # TgC
    
    sd_mask_yr = (SWEI <= -1.3).rename({"time": "year"}).load()

    # --- growing season ---
    swc_start_month = (swe_month + 1).clip(max=9)
    month_ok = xr.apply_ufunc(
        lambda m, start: (m >= start) & (m <= 9),
        gpp_anom["time.month"], swc_start_month,
        input_core_dims=[[], []],
        vectorize=True, dask="parallelized", output_dtypes=[bool],)
    
    gpp_season = gpp_anom.where(month_ok, drop=False).load()
    gpp_yearly = gpp_season.groupby("time.year").sum(dim="time", skipna=True, min_count=1)
    
    sd_sum     = gpp_yearly.where(sd_mask_yr).sum(dim="year", skipna=True, min_count=1)
    nosd_sum   = gpp_yearly.where(~sd_mask_yr).sum(dim="year", skipna=True, min_count=1)
    
    sd_count   = sd_mask_yr.sum(dim="year")
    nosd_count = (~sd_mask_yr).sum(dim="year")
    
    sd_mean    = sd_sum.where(sd_count > 0) / sd_count.where(sd_count > 0)
    nosd_mean  = nosd_sum.where(nosd_count > 0) / nosd_count.where(nosd_count > 0)
    
    diff_data[case] = sd_mean - nosd_mean


    # --- category masks ---
    #=== SWC masks ============================================================================================
    swc_dry  = xr.open_dataset(out_dir + "mask_swc_dry.nc")["swc_vol"].rename({'time': 'year'})
    swc_norm = xr.open_dataset(out_dir + "mask_swc_normal.nc")["swc_vol"].rename({'time': 'year'})
    swc_wet  = xr.open_dataset(out_dir + "mask_swc_wet.nc")["swc_vol"].rename({'time': 'year'})
    
    #=== SWC masks ============================================================================================
    swc_summer_dry  = xr.open_dataset(out_dir + "mask_swc_summer_dry.nc")[swc_var].drop_vars('year', errors='ignore').rename({'time': 'year'})
    swc_summer_norm = xr.open_dataset(out_dir + "mask_swc_summer_normal.nc")[swc_var].drop_vars('year', errors='ignore').rename({'time': 'year'})
    swc_summer_wet  = xr.open_dataset(out_dir + "mask_swc_summer_wet.nc")[swc_var].drop_vars('year', errors='ignore').rename({'time': 'year'})

    # === Spring SPI masks ============================================================================================
    spi_spring_neg  = xr.open_dataset(out_dir + "mask_spi_spring_-.nc")["spi_gamma_03"].transpose('time','lat','lon').rename({'time': 'year'})
    spi_spring_zero = xr.open_dataset(out_dir + "mask_spi_spring_0.nc")["spi_gamma_03"].transpose('time','lat','lon').rename({'time': 'year'})
    spi_spring_pos  = xr.open_dataset(out_dir + "mask_spi_spring_+.nc")["spi_gamma_03"].transpose('time','lat','lon').rename({'time': 'year'})
    
    # === Summer SPI masks ============================================================================================
    spi_summer_neg  = xr.open_dataset(out_dir + "mask_spi_summer_-.nc")["spi_gamma_03"].transpose('time','lat','lon').rename({'time': 'year'})
    spi_summer_zero = xr.open_dataset(out_dir + "mask_spi_summer_0.nc")["spi_gamma_03"].transpose('time','lat','lon').rename({'time': 'year'})
    spi_summer_pos  = xr.open_dataset(out_dir + "mask_spi_summer_+.nc")["spi_gamma_03"].transpose('time','lat','lon').rename({'time': 'year'})
    
    
    cat_masks = {
        'Spring SWC':       {'dry': swc_dry,       'normal': swc_norm,       'wet': swc_wet},
        'Summer SWC': {'dry': swc_summer_dry, 'normal': swc_summer_norm, 'wet': swc_summer_wet},
        'Spring Precip':    {'dry': spi_spring_neg, 'normal': spi_spring_zero,'wet': spi_spring_pos},
        'Summer Precip':    {'dry': spi_summer_neg, 'normal': spi_summer_zero,'wet': spi_summer_pos},
    }
    
    cat_masks = {
        'Spring SWC':    {'dry': swc_dry,        'normal': swc_norm,        'wet': swc_wet},
        'Summer SWC':    {'dry': swc_summer_dry, 'normal': swc_summer_norm, 'wet': swc_summer_wet},
        'Spring Precip': {'dry': spi_spring_neg,  'normal': spi_spring_zero, 'wet': spi_spring_pos},
        'Summer Precip': {'dry': spi_summer_neg,  'normal': spi_summer_zero, 'wet': spi_summer_pos},
        'Summer Precip with normal spring': {
            'dry':    spi_spring_zero.astype(bool) & spi_summer_neg.astype(bool),
            'normal': spi_spring_zero.astype(bool) & spi_summer_zero.astype(bool),
            'wet':    spi_spring_zero.astype(bool) & spi_summer_pos.astype(bool),
        },
    }
    

    # --- compute diff maps per var/cat ---
    for var_name in var_names:
        
        if var_name == 'Spring SWC':
            month_indices = get_swc_indices(swe_month)
            
        elif var_name == 'Spring Precip':
            month_indices = spring_idx
            
        elif var_name == "Summer SWC":
            month_indices = summer_swc_idx
            
        elif var_name == "Summer Precip with normal spring":
            month_indices = summer_idx
            
        else:
            month_indices = summer_idx

        for cat in categories:
            diff_maps_cat[var_name][cat][case] = compute_diff_map(gpp_season, sd_mask_yr, cat_masks[var_name][cat],valid_mask=valid_land_case, month_indices=month_indices) #threshold_mask


# === REFERENCE PLOTS (no category) =====================================
diff_stack    = xr.concat([diff_data[c] for c in cases], dim="case") #stacking the 4 datasets
diff_mean     = diff_stack.mean(dim="case", skipna=True)
n_negative    = (diff_stack < 0).sum(dim="case")
n_positive    = (diff_stack > 0).sum(dim="case")
diff_agreed_3 = diff_mean.where((n_negative >= 3) | (n_positive >= 3))
diff_input    = diff_agreed_3


# === CATEGORY BAR PLOTS ================================================
eco_order           = [13, 8, 4, 10, 5, 6, 11]

titles_cat = [
    "Spring SWC condition (months SWE peak + 1)", #{swc_offset_before} to SWE peak + {swc_offset_after}
    f"Summer SWC condition (months {summer_swc_start}–{summer_swc_end})",
    f"Spring Precip condition (months {spring_start}–{spring_end})",
    f"Summer Precip condition (months {summer_start}–{summer_end})",
    f"Summer Precip (normal spring filter) (months {summer_start}–{summer_end})"]


for var_name, title_cat in zip(var_names, titles_cat):
    
    # --- Build labels with correct var_name and merged groups ---
    eco_ylabels = []
    for gid in eco_order_merged:
        counts = []
        for cat in categories:
            case_maps  = [diff_maps_cat[var_name][cat][c] for c in cases]
            diff_stack = xr.concat(case_maps, dim="case")
            diff_mean  = diff_stack.mean(dim="case", skipna=True)
            n_neg      = (diff_stack < 0).sum(dim="case")
            n_pos      = (diff_stack > 0).sum(dim="case")
            diff_input = diff_mean.where((n_neg >= 3) | (n_pos >= 3))
            vals = diff_input.where(get_group_mask(eco_map, gid)).values.ravel()
            vals = vals[~np.isnan(vals)]
            counts.append(len(vals))
        name = eco_merge_labels.get(gid, ecolabels.get(gid, str(gid)))
        eco_ylabels.append(f"{name}\nD:{counts[0]} N:{counts[1]} W:{counts[2]}")

    ai_ylabels = []
    for aid in ai_order_merged:
        counts = []
        for cat in categories:
            case_maps  = [diff_maps_cat[var_name][cat][c] for c in cases]
            diff_stack = xr.concat(case_maps, dim="case")
            diff_mean  = diff_stack.mean(dim="case", skipna=True)
            n_neg      = (diff_stack < 0).sum(dim="case")
            n_pos      = (diff_stack > 0).sum(dim="case")
            diff_input = diff_mean.where((n_neg >= 3) | (n_pos >= 3))
            vals = diff_input.where(get_group_mask(ai_map, aid)).values.ravel()
            vals = vals[~np.isnan(vals)]
            counts.append(len(vals))
        name = ai_merge_labels.get(aid, ai_labels.get(aid, str(aid)))
        ai_ylabels.append(f"{name}\nD:{counts[0]} N:{counts[1]} W:{counts[2]}")

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5),sharey=True)
    #fig.text(0.07, 0.93,title_cat,fontsize=12, va="top", ha="left",bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85),)
    #fig.suptitle(f'∆GPP (≥3/4 agree) by {title_cat} \n(bar=mean, whiskers=IQR, line=median)',fontsize=13)
        
    for ax, (group_name, group_ids, group_labels, group_map) in zip(axes, [
        ("Ecoregion", eco_order_merged, eco_ylabels, eco_map),
        ("Aridity Index Class",   ai_order_merged,  ai_ylabels,  ai_map),]):

        bar_width = 0.25
        x = np.arange(len(group_ids))
        
        #pixel_counts = {cat: [] for cat in categories} #store pixel count

        for k, cat in enumerate(categories):

            # stack 4 cases → mean + agreement mask
            case_maps  = [diff_maps_cat[var_name][cat][c] for c in cases]
            diff_stack = xr.concat(case_maps, dim="case")
            diff_mean  = diff_stack.mean(dim="case", skipna=True)
            n_neg      = (diff_stack < 0).sum(dim="case")
            n_pos      = (diff_stack > 0).sum(dim="case")
            diff_input = diff_mean.where((n_neg >= 3) | (n_pos >= 3))

            medians, iqr_low, iqr_high, means = [], [], [], []
            

            for gid in group_ids:
                #mask = (group_map == gid)
                mask = get_group_mask(group_map, gid)
                vals = diff_input.where(mask).values.ravel()
                vals = vals[~np.isnan(vals)]

                if len(vals) > 0:
                    q25, q75 = np.percentile(vals, [25, 75])
                    medians.append(np.median(vals))
                    means.append(np.mean(vals))
                    #iqr_low.append(np.median(vals) - q25)
                    #iqr_high.append(q75 - np.median(vals))
                    iqr_low.append(q25)
                    iqr_high.append(q75)
                    
                else:
                    medians.append(np.nan)
                    means.append(np.nan)
                    iqr_low.append(np.nan)
                    iqr_high.append(np.nan)


            medians  = np.array(medians)
            iqr_low  = np.array(iqr_low)
            iqr_high = np.array(iqr_high)            
            offset = (k - 1) * bar_width

            # Draw bars without yerr
            ax.bar(x + offset, means, bar_width,
                   label=cat_labels[cat],
                   color=cat_colors[cat],
                   alpha=0.8,edgecolor='k',linewidth=0.5)
            
            # Draw IQR whiskers manually, centered on median
            for i in range(len(group_ids)):
                xpos = x[i] + offset
                ax.plot([xpos, xpos], [iqr_low[i], iqr_high[i]], color='black', linewidth=0.8, solid_capstyle='round')
                # end caps
                ax.plot([xpos - 0.05, xpos + 0.05], [iqr_low[i], iqr_low[i]],color='black', linewidth=0.8)
                ax.plot([xpos - 0.05, xpos + 0.05], [iqr_high[i], iqr_high[i]],color='black', linewidth=0.8)
            
            # Median line
            ax.scatter(x + offset, medians,color='black', s=100, marker='_',zorder=5, label='_nolegend_')

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=35, ha='right', fontsize=font_leg)
        ax.set_ylabel(f' ∆GPP {gpp_units}', fontsize=font_title)
        ax.set_title(group_name, fontsize=font_title)
        ax.tick_params(axis='y', labelsize=font_leg)
        ax.legend(fontsize=font_leg)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    

'''
###################
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_condition_map(var_name, cat, group_map, group_id, group_label,
                       sign=None, title_extra=""):
    """
    Maps pixels for a given var/category/group.
    sign: None = show all agreed pixels coloured by diff value
          'pos' = highlight only positive diff pixels
          'neg' = highlight only negative diff pixels
    """
    case_maps  = [diff_maps_cat[var_name][cat][c] for c in cases]
    diff_stack = xr.concat(case_maps, dim="case")
    diff_mean  = diff_stack.mean(dim="case", skipna=True)
    n_neg      = (diff_stack < 0).sum(dim="case")
    n_pos      = (diff_stack > 0).sum(dim="case")
    diff_input = diff_mean.where((n_neg >= 3) | (n_pos >= 3))

    # mask to the group region
    region_mask = (group_map == group_id)
    diff_region = diff_input.where(region_mask)

    # optionally filter by sign
    if sign == 'pos':
        diff_region = diff_region.where(diff_region > 0)
    elif sign == 'neg':
        diff_region = diff_region.where(diff_region < 0)

    fig, ax = plt.subplots(figsize=(12, 5),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3)
    ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=0)

    vmax = float(np.nanpercentile(np.abs(diff_input.values), 95))
    im = diff_region.plot(ax=ax, transform=ccrs.PlateCarree(),cmap='RdBu', vmin=-vmax, vmax=vmax,add_colorbar=True,cbar_kwargs={'label': f'SD−noSD GPP {gpp_units}','shrink': 0.6})

    sign_str = {"pos": " (positive only)", "neg": " (negative only)",
                None: ""}.get(sign, "")
    ax.set_title(f'{var_name} | {cat_labels[cat]} | {group_label}{sign_str}\n{title_extra}',
                 fontsize=11)
    ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())
    plt.tight_layout()
    plt.show()


# --- calls ---
plot_condition_map('Spring SWC', 'dry',group_map=eco_map, group_id=8, group_label='Temperate Grasslands, Savannas & Shrublands',sign='pos')
plot_condition_map('Spring Precip', 'dry',group_map=eco_map, group_id=8, group_label='Temperate Grasslands, Savannas & Shrublands',sign='pos')
plot_condition_map('Summer Precip', 'dry',group_map=eco_map, group_id=8, group_label='Temperate Grasslands, Savannas & Shrublands',sign='pos')

plot_condition_map('Spring SWC', 'dry',group_map=ai_map, group_id=2, group_label='Arid',sign='pos')
plot_condition_map('Spring Precip', 'dry',group_map=ai_map, group_id=2, group_label='Arid',sign='pos')
plot_condition_map('Summer Precip', 'dry',group_map=ai_map, group_id=2, group_label='Arid',sign='pos')

plot_condition_map('Spring SWC', 'dry',group_map=ai_map, group_id=5, group_label='humid',sign='neg')
plot_condition_map('Spring Precip', 'dry',group_map=ai_map, group_id=5, group_label='humid',sign='neg')
plot_condition_map('Summer Precip', 'dry',group_map=ai_map, group_id=5, group_label='humid',sign='neg')
'''

