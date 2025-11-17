import numpy as np
from scipy.spatial import cKDTree
import datetime
from joblib import Parallel, delayed, parallel_backend
import wradlib as wrl
import xradar as xd
import argparse
import os
import rioxarray as rxr

def correct_radar_composites(starttime_str):
    moments = {}
    datet_orig = datetime.datetime.strptime(starttime_str, "%Y%m%d%H%M")
    datet = datet_orig
    # Open first set of files
    try:
        first_times = np.load("/home/users/myllykos/mepsi/MEPS_advection_based_adjustment_of_radar_measurements/correction_maps_composite/"+starttime_str+"_time.npz")['arr_0']
        
        for t_i in range(int(first_times[~np.isnan(first_times)].max())+1):
            timestamp = datetime.datetime.strftime(datet,"%Y%m%d%H%M")
            filename = "/arch/radar/storage/"+str(datet.year)+"/"+str(datet.month).zfill(2)+"/"+str(datet.day).zfill(2)+"/fmi/radar/iris/GeoTIFF/"+timestamp+"_SUOMI250_FIN.tif"
        
            try:
                with rxr.open_rasterio(filename) as ds:
                    moments[datet] = ds.data

                    if t_i == 0:
                        vali = ds
                        current_to_be_corrected = vali
            except:
                
                if t_i == 0:
                    varapaiva =datetime.datetime(2025,11,11,4,0)
                    vara_timestamp = datetime.datetime.strftime(varapaiva,"%Y%m%d%H%M")
                    filename = "/arch/radar/storage/"+str(varapaiva.year)+"/"+str(varapaiva.month).zfill(2)+"/"+str(varapaiva.day).zfill(2)+"/fmi/radar/iris/GeoTIFF/"+vara_timestamp+"_SUOMI250_FIN.tif"
                    with rxr.open_rasterio(filename) as ds:
                        vali = ds
        
                    current_to_be_corrected = vali
                    
                moments[datet] = np.zeros(vali.shape)*np.nan
                with open("korjaus_failaa_composite", 'a', encoding='utf-8') as file:
                    file.write(timestamp + '\n')
            datet-=datetime.timedelta(minutes=5)
        first_times = None
        # Iterate over the dataset
        end_time = datet_orig + datetime.timedelta(hours=3)
        
        current_time = datet_orig
        while current_time <= end_time:
            curtime_str = current_time.strftime("%Y%m%d%H%M")
            print("Now correcting:",curtime_str) 
            final_lat = np.load("/home/users/myllykos/mepsi/MEPS_advection_based_adjustment_of_radar_measurements/correction_maps_composite/"+curtime_str+"_lat.npz")['arr_0']
            final_lon = np.load("/home/users/myllykos/mepsi/MEPS_advection_based_adjustment_of_radar_measurements/correction_maps_composite/"+curtime_str+"_lon.npz")['arr_0']
            final_times = np.load("/home/users/myllykos/mepsi/MEPS_advection_based_adjustment_of_radar_measurements/correction_maps_composite/"+curtime_str+"_time.npz")['arr_0']

            last_in_past = int(final_times[~np.isnan(final_times)].max())
            datet = current_time
            mapping_to_timestamp = {}
            for t_i in range(last_in_past+1):
                if not datet in moments.keys():
                    timestamp = datetime.datetime.strftime(datet,"%Y%m%d%H%M")
                    print(timestamp, flush=True)
                    filename = "/arch/radar/storage/"+str(datet.year)+"/"+str(datet.month).zfill(2)+"/"+str(datet.day).zfill(2)+"/fmi/radar/iris/GeoTIFF/"+timestamp+"_SUOMI250_FIN.tif"
                    try:
                        with rxr.open_rasterio(filename) as ds:
                            moments[datet] = ds.data
                    except:
                        moments[datet] = np.zeros(vali.shape)*np.nan
                mapping_to_timestamp[t_i] = datet
                datet-=datetime.timedelta(minutes=5)

            def collect_correction(az,r):
                if ~np.isnan(final_times[az,r]):
                    arra = moments[mapping_to_timestamp[max(0,final_times[az,r])]][0,az,r]
                    return arra
                return np.nan
            #print("toimii0",flush=True)
            
            #with parallel_backend("loky", inner_max_num_threads=2):
            #    data = Parallel(n_jobs=4,verbose=1)(delayed(collect_correction)(i,j)  for i in range(0,360) for j in range(0,500))
            
            data = Parallel(n_jobs=1,verbose=1)(delayed(collect_correction)(i,j)  for i in range(0,vali.shape[1]) for j in range(0,vali.shape[2]))
            
            #print("toimii1",flush=True)
            advection_corrected_dbz = np.zeros(vali.shape)*np.nan

            year = current_time.year
            month = current_time.month
            day = current_time.day
            dir_path = os.path.join('corrected_field_composite', str(year), str(month).zfill(2), str(day).zfill(2))
            
            #for az in range(vali.shape[1]):
            #    for r in range(vali.shape[2]):
            #        advection_corrected_dbz[0,az,r] = data[az*vali.shape[2]+r]
            
            advection_corrected_dbz[0, :, :] = np.array(data).reshape(vali.shape[1], vali.shape[2])
            
            advection_corrected_dbz[np.isnan(advection_corrected_dbz)] = -500
            
            single_sweep_data = current_to_be_corrected
            vali = single_sweep_data
            root = current_to_be_corrected.copy()
            
            root.data = advection_corrected_dbz#, dims=single_sweep_data.dims, coords=single_sweep_data.coords)
            root.attrs['_Undetect'] = -500
            
            year = current_time.year
            month = current_time.month
            day = current_time.day
            dir_path = os.path.join('corrected_composite/', str(year), str(month).zfill(2), str(day).zfill(2))
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, 'wind_drift_corrected_'+curtime_str+'.tif')
            compression = {"compress": "lzw", "zlevel": 9} 
            root.rio.to_raster(file_path, encoding = {root.name: compression})

            current_time += datetime.timedelta(minutes=5)
    except:
        print("koko failas")
        with open("mappays_failaa_composite_alkaen_", 'a', encoding='utf-8') as file:
            file.write(starttime_str + '\n')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute tif files using the wind drift correction mappings created in compute_wind_drift_correction_composite.py"
    )
    parser.add_argument("starttime", help="start time (YYYYmmddHHMM)")
    #parser.add_argument("config", help="configuration profile to use")
    args = parser.parse_args()
    starttime = args.starttime

    correct_radar_composites(starttime)

    # Define the path to the directory containing the files
    directory_path = '/home/users/myllykos/mepsi/MEPS_advection_based_adjustment_of_radar_measurements/correction_maps_composite'

    # Loop through each file in the directory
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        
        # Check if it's a file and has a .npy extension
        #if os.path.isfile(file_path) and file.endswith('.npy'):
        #    try:
        #        # Delete the .npy file
        #        os.remove(file_path)
        #        print(f"Deleted: {file_path}")
        #    except Exception as e:
        #        print(f"Error deleting file {file_path}: {e}")
