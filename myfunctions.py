import numpy as np
import xarray as xr
from scipy.stats import norm
from scipy.stats import mode

#calculate anomalies
def anomalies(data): 
    # Group by month, compute the mean for each month across all years
    month = data.groupby("time.month")
    month_avg = month.mean(dim="time")
    #subtract monthly mean from each corresponding month   
    anomalies = month - month_avg
    
    return anomalies

#calculate snow threshold mask
def threshold_mask(swe,name):
##Look at 3-month windows (12-1-2, 11-12-1, 2-3-4, etc.)
##Look at all years per gridcell, include gridcell if SWE>0 in the 3-month window 75% of the time
    
    # Step 1: rolling 3-month sum
    
    #method to be able to get windows like "dec-jan-feb"
    swe_prev = swe.shift(time=1)  # SWE for the previous month
    swe_next = swe.shift(time=-1)  # SWE for the next month
    swe_3mo = swe + swe_prev + swe_next
    
    # Step 2: Check for nonzero SWE
    swe_nonzero = swe_3mo > 0  # Boolean mask where SWE > 0
    
    # Count how many months per year have nonzero SWE value
    valid_counts = swe_nonzero.groupby("time.year").sum(dim="time")
    
    #Convert valid_counts to a binary mask (1 if any valid month, 0 otherwise)
    valid_years = (valid_counts > 0).astype(int)
    
    # Step 3: Count the number of valid years per grid cell
    valid_year_count = valid_years.sum(axis=0)  # Now shape is (lat, lon)
    
    #defining the threshold of 75% of the years 
    years = np.unique(swe_3mo["time.year"])  #number of years
    threshold = int(0.75 * len(years))  # 75% of available years
    
    threshold_mask = (valid_year_count >= threshold).astype(int)  # 1 = included, 0 = excluded
    #final_mask.to_netcdf("C:/Users/m.varela/Documents/Data_SnowDrought/REAL_data_analysis/fixed_percentiles/"+case+"/"+name+".nc")

    return threshold_mask 


def swei_1d(ts, months):
    #---------------------------------------
    #SWEI computation for 1D time series 
    #---------------------------------------
    result = np.full_like(ts, np.nan, dtype=float)

    for month in range(1, 13):

        mask = (months == month) & (~np.isnan(ts))
        vals = ts[mask]

        N = len(vals)

        sorted_vals = np.sort(vals)
        ranks = np.searchsorted(sorted_vals, vals) + 1

        P = (ranks - 0.44) / (N + 0.12)

        result[mask] = norm.ppf(P)

    return result


def compute_swei(swe, window=3):
    
    #------------------------------
    # Compute 3-month rolling sum 
    #------------------------------
    swe_roll = swe.rolling(time=window, min_periods=window, center=False).sum()

    swei = xr.apply_ufunc(
        swei_1d,
        swe_roll,
        swe_roll["time.month"],
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    swei.name = f"SWEI_{window}month"

    return swei

def calculate_mode(da):
    """
    Calculate the modal month (most frequent) along time dimension.
    """
    def _mode_1d(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        vals, counts = np.unique(x.astype(int), return_counts=True)
        return vals[np.argmax(counts)]

    return xr.apply_ufunc(
        _mode_1d,
        da,
        input_core_dims=[["time"]],
        vectorize=True,
        output_dtypes=[float])