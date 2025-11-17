import numpy as np
from scipy.spatial import cKDTree
import datetime
from joblib import Parallel, delayed, parallel_backend
import wradlib as wrl
import xradar as xd
import argparse
import os


def correct_radar_volumes(starttime_str, field = 'DBZH'):
    moments = {}
    datet_orig = datetime.datetime.strptime(starttime_str, "%Y%m%d%H%M")
    datet = datet_orig
    try:
        first_times = np.load("advektiokorjaus/correction_maps_first/"+starttime_str+"_time.npy")
    
        for t_i in range(int(first_times[~np.isnan(first_times)].max())+1):
            timestamp = datetime.datetime.strftime(datet,"%Y%m%d%H%M")
            filename = "/arch/radar/HDF5/"+str(datet.year)+"/"+str(datet.month).zfill(2)+"/"+str(datet.day).zfill(2)+"/radar/polar/fiuta/"+timestamp+"_radar.polar.fiuta.h5"
            try:
                pvol = xd.io.open_odim_datatree(filename)
                if t_i == 0:
                    current_to_be_corrected = pvol
            
                    vali = pvol["sweep_0"].ds.wrl.georef.georeference(
                        crs=wrl.georef.get_default_projection()
                    )     
                    moments[datet] = vali[field].data
                else:
                    moments[datet] = pvol["sweep_0"].ds.wrl.georef.georeference(
                        crs=wrl.georef.get_default_projection()
                    )[field].data
                
                # Tarkista ettei ole väärän muotoinen
                if moments[datet].shape != (360,500):
                    moments[datet] = np.zeros((360,500))*np.nan
                    with open("korjaus_failaa_" + field, 'a', encoding='utf-8') as file:
                        file.write(timestamp + '\n')
            except:
                if t_i == 0:
                    varapaiva =datetime.datetime(2024,11,21,4,0)
                    vara_timestamp = datetime.datetime.strftime(varapaiva,"%Y%m%d%H%M")

                    filename = "/arch/radar/HDF5/"+str(varapaiva.year)+"/"+str(varapaiva.month).zfill(2)+"/"+str(varapaiva.day).zfill(2)+"/radar/polar/fiuta/"+vara_timestamp+"_radar.polar.fiuta.h5"
                    pvol = xd.io.open_odim_datatree(filename)
                
                    vali = pvol["sweep_0"].ds.wrl.georef.georeference(
                        crs=wrl.georef.get_default_projection()
                    )     
                    current_to_be_corrected = pvol
                    
                moments[datet] = np.zeros((360,500))*np.nan
                with open("korjaus_failaa", 'a', encoding='utf-8') as file:
                    file.write(timestamp + '\n')
            datet-=datetime.timedelta(minutes=5)
        first_times = None
        vali = vali.to_dataarray()
        
        cKDTree_radar = cKDTree(np.column_stack((vali.y.values.flatten(), vali.x.values.flatten())))
        
        def get_closest_radar_coordinate(x,y):
            dist, idx = cKDTree_radar.query((y, x))
            return np.unravel_index(idx, (360,500))
        
        end_time = datet_orig + datetime.timedelta(hours=3)
        current_time = datet_orig
        while current_time <= end_time:
            curtime_str = current_time.strftime("%Y%m%d%H%M")
            print("Now correcting:",curtime_str) 
            final_lat = np.load("advektiokorjaus/correction_maps_first/"+curtime_str+"_lat.npy")
            final_lon = np.load("advektiokorjaus/correction_maps_first/"+curtime_str+"_lon.npy")
            final_times = np.load("advektiokorjaus/correction_maps_first/"+curtime_str+"_time.npy")

            last_in_past = int(final_times[~np.isnan(final_times)].max())
            datet = current_time
            mapping_to_timestamp = {}
            for t_i in range(last_in_past+1):
                if not datet in moments.keys():
                    timestamp = datetime.datetime.strftime(datet,"%Y%m%d%H%M")
                    print(timestamp, flush=True)
                    filename = f"/arch/radar/HDF5/"+str(datet.year)+"/"+str(datet.month).zfill(2)+"/"+str(datet.day).zfill(2)+"/radar/polar/fiuta/"+timestamp+"_radar.polar.fiuta.h5"
                    try:
                        pvol = xd.io.open_odim_datatree(filename)
                        
                        moments[datet] = pvol["sweep_0"].ds.wrl.georef.georeference(
                            crs=wrl.georef.get_default_projection()
                        )[field].data
                    except:
                        moments[datet] = np.zeros((360,500))*np.nan
                mapping_to_timestamp[t_i] = datet
                datet-=datetime.timedelta(minutes=5)

            def collect_correction(az,r):
                if ~np.isnan(final_times[az,r]):
                    
                    az_radar,r_radar = get_closest_radar_coordinate(final_lon[az,r],final_lat[az,r])
                    return moments[mapping_to_timestamp[max(0,final_times[az,r])]][az_radar,r_radar]
                return np.nan
            #print("toimii0",flush=True)
            
            #with parallel_backend("loky", inner_max_num_threads=2):
            #    data = Parallel(n_jobs=4,verbose=1)(delayed(collect_correction)(i,j)  for i in range(0,360) for j in range(0,500))
            
            data = Parallel(n_jobs=1,verbose=1)(delayed(collect_correction)(i,j)  for i in range(0,360) for j in range(0,500))
            
            #print("toimii1",flush=True)
            advection_corrected_dbz = np.zeros((360,500))*np.nan
            for az in range(360):
                for r in range(500):
                    advection_corrected_dbz[az][r] = data[az*500+r]
            #print("toimii2",flush=True)
            advection_corrected_dbz[np.isnan(advection_corrected_dbz)] = -500
            
            
            
            
            
            
            single_sweep_data = current_to_be_corrected['sweep_0']
            vali = single_sweep_data[field]
            root = current_to_be_corrected.copy()
            #print("toimii3",flush=True)
            single_sweep_data[field].data = advection_corrected_dbz#, dims=single_sweep_data.dims, coords=single_sweep_data.coords)
            single_sweep_data[field].attrs['_Undetect'] = -500
            #print("toimii4",flush=True)
            list_of_to_be_removed = ["CSP","VRADH","WRADH","ZDR","KDP","RHOHV","DBZHC","VRADDH","TH","SQIH","PHIDP","HCLASS","ZDRC","TV","SNR","PMI","DBZV","DBZH"]
            list_of_to_be_removed = [item for item in list_of_to_be_removed if item != field]
            #print("toimii5",flush=True)
            for var in list_of_to_be_removed:
                single_sweep_data[var] = None
            
            root._children.clear()
            root._children["sweep_0"]= single_sweep_data

            #print("toimii6",flush=True)
            year = current_time.year
            month = current_time.month
            day = current_time.day
            dir_path = os.path.join('corrected_field_'+field+'/', str(year), str(month).zfill(2), str(day).zfill(2))
            os.makedirs(dir_path, exist_ok=True)
            #print("toimii7",flush=True)
            file_path = os.path.join(dir_path, curtime_str+'.h5')

            xd.io.to_odim(root, file_path, source = "WIGOS:0-246-0-101872,WMO:02870,RAD:FI47,PLC:Utajaervi,NOD:fiuta")        
            current_time += datetime.timedelta(minutes=5)
    except:
        with open("mappays_failaa_alkaen_"+field, 'a', encoding='utf-8') as file:
            file.write(starttime_str + '\n')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute ODIM h5 files using the advection correction mappings created in compute_advection_correction.py"
    )
    parser.add_argument("starttime", help="start time (YYYYmmddHHMM)")
    #parser.add_argument("config", help="configuration profile to use")
    args = parser.parse_args()
    starttime = args.starttime

    correct_radar_volumes(starttime)

    # Define the path to the directory containing the files
    directory_path = '/home/users/myllykos/mepsi/MEPS_advection_based_adjustment_of_radar_measurements/correction_maps/'  # Change this to your actual directory path

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
