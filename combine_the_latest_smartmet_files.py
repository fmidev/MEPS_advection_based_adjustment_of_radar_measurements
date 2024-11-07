import xarray as xr
import numpy as np

def combine_grib_files(latest, old):
    if np.min(latest.time.data) <= np.max(old.time.data):
        raise ValueError(f"Latest time of model must be greater than old time of model!")
    latest_valid_time = latest.valid_time.data
    old_valid_time = old.valid_time.data
    time_diff = np.min(latest.time.data-old.time.data)
    intersection = np.intersect1d(latest_valid_time,old_valid_time)
    #print(intersection)
    if len(intersection) > 0:
        
        mask = ~np.isin(old_valid_time, intersection)
        non_overlap_values = old.step.data[mask]
        # ongelma nyt on se, että step on volyymikohtainen
        #print(np.max(old.step.data))
        #print((latest.step+.data)
        old = old.sel({'step': non_overlap_values})
        new_step = latest.step.data + time_diff
        latest = latest.assign_coords(step=new_step, time=old.time.data)
        
    
    combined_ds = xr.concat([latest, old], dim='step').sortby('step')
    combined_ds.time.data = np.max(old.time.data)
    
    return combined_ds

ds1 = xr.open_dataset('smartmets/latest_smartmet.grib', engine='cfgrib')
ds2 = xr.open_dataset('smartmets/second_latest_smartmet.grib', engine='cfgrib')
ds3 = xr.open_dataset('smartmets/oldest_smartmet.grib', engine='cfgrib')

combined_ds = combine_grib_files(ds2, ds3)
combined_ds = combine_grib_files(ds1, combined_ds)
print(combined_ds.step.data)
# Save the combined dataset if needed
combined_ds.to_netcdf('smartmets/combined_data.nc')
