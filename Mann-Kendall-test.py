import xarray as xr
import numpy as np
import pymannkendall as mk
from scipy import stats

# === datasets =================================================================================================
cases = {
    "fluxcom" : "D:/ra25nir/Data_complete/fluxcom_detrended.nc",
    "gosif"   : "D:/ra25nir/Data_complete/gosif_detrended.nc",
    "vodca2"  : "D:/ra25nir/Data_complete/vodca2_detrended.nc",
    "nirv"    : "D:/ra25nir/Data_complete/NIRv_detrended.nc",
    "lpjml"   : "C:/Users/m.varela/Documents/Data_SnowDrought/ERA5L_LPJmL_outputs/gpp_detrended.nc",
}

var_names = {"fluxcom" : "GPP","gosif"   : "GPP", "vodca2"  : "GPP","nirv"    : "GPP","lpjml"   : "GPP",}

# === FUNCTION: spatial mean → MK test ===========================================================================
def run_mk_test(file_path, var_name, case_name):
    print(f"\n{'='*55}")
    print(f"  Dataset : {case_name}")
    print(f"  File    : {file_path}")
    print(f"{'='*55}")

    # Load data
    ds = xr.open_dataset(file_path)
    da = ds[var_name]

    # Collapse to global mean time series (ignoring NaNs)
    ts = da.mean(dim=[d for d in da.dims if d != "time"], skipna=True).values

    # Drop NaNs just in case
    ts = ts[~np.isnan(ts)]

    # ── Mann-Kendall test ──
    result = mk.original_test(ts)
    
    # Add relative trend assessment
    ts_mean = np.mean(ts)
    ts_std  = np.std(ts)
    relative_slope = (result.slope / ts_mean) * 100  # % per time step

    print(f"  MK Trend       : {result.trend}")        # increasing / decreasing / no trend
    print(f"  p-value        : {result.p:.4f}")
    print(f"  Kendall's tau  : {result.Tau:.4f}")
    print(f"  Sen's slope    : {result.slope:.6f}")     # magnitude of remaining trend
    print(f"  Time series mean   : {ts_mean:.4f}")
    print(f"  Time series std    : {ts_std:.4f}")
    print(f"  Relative slope     : {relative_slope:.4f} % per timestep")
    print(f"  Slope / Std ratio  : {abs(result.slope) / ts_std:.4f}  (< 0.01 = negligible)")
    print(f"  Significance   : {'✅ No trend (well detrended)' if result.p > 0.05 else '⚠️  Trend still present!'}")

    return result

# === RUN =========================================================================================================
results = {}
for case_name, file_path in cases.items():
    results[case_name] = run_mk_test(file_path, var_names[case_name], case_name)