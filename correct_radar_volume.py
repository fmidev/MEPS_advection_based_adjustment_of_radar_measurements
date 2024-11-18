import numpy as np
from scipy.spatial import cKDTree
import datetime
from joblib import Parallel, delayed, parallel_backend
import wradlib as wrl
import xradar as xd
import argparse


def correct_radar_volumes(starttime_str):
    moments = {}
    datet_orig = datetime.datetime.strptime(starttime_str, "%Y%m%d%H%M")
    datet = datet_orig
    first_times = np.load("correction_maps/"+starttime_str+"_time.npy")
    
    for t_i in range(int(first_times[~np.isnan(first_times)].max())+1):
        timestamp = datetime.datetime.strftime(datet,"%Y%m%d%H%M")
        filename = "/arch/radar/HDF5/"+str(datet.year)+"/"+str(datet.month).zfill(2)+"/"+str(datet.day).zfill(2)+"/radar/polar/fivih/"+timestamp+"_radar.polar.fivih.h5"
        
        pvol = xd.io.open_odim_datatree(filename)
        if t_i == 0:
            vali = pvol["sweep_0"].ds.wrl.georef.georeference(
                crs=wrl.georef.get_default_projection()
            )     
            moments[datet] = vali['DBZH'].data
        else:
            moments[datet] = pvol["sweep_0"].ds.wrl.georef.georeference(
                crs=wrl.georef.get_default_projection()
            )['DBZH'].data
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
        final_lat = np.load("correction_maps/"+curtime_str+"_lat.npy")
        final_lon = np.load("correction_maps/"+curtime_str+"_lon.npy")
        final_times = np.load("correction_maps/"+curtime_str+"_time.npy")

        last_in_past = int(final_times[~np.isnan(final_times)].max())
        datet = current_time
        mapping_to_timestamp = {}
        for t_i in range(last_in_past+1):
            
            if not datet in moments.keys():
                timestamp = datetime.datetime.strftime(datet,"%Y%m%d%H%M")
                print(timestamp, flush=True)
                filename = f"/arch/radar/HDF5/"+str(datet.year)+"/"+str(datet.month).zfill(2)+"/"+str(datet.day).zfill(2)+"/radar/polar/fivih/"+timestamp+"_radar.polar.fivih.h5"
                
                pvol = xd.io.open_odim_datatree(filename)
                if t_i == 0:
                    current_to_be_corrected = pvol
                moments[datet] = pvol["sweep_0"].ds.wrl.georef.georeference(
                    crs=wrl.georef.get_default_projection()
                )['DBZH'].data
            mapping_to_timestamp[t_i] = datet
            datet-=datetime.timedelta(minutes=5)

        def collect_correction(az,r):
            if ~np.isnan(final_times[az,r]):
                az_radar,r_radar = get_closest_radar_coordinate(final_lon[az,r],final_lat[az,r])
                return moments[mapping_to_timestamp[final_times[az,r]]][az_radar,r_radar]
            return np.nan

        with parallel_backend("loky", inner_max_num_threads=2):
            data = Parallel(n_jobs=4,verbose=1)(delayed(collect_correction)(i,j)  for i in range(0,360) for j in range(0,500))
        
        advection_corrected_dbz = np.zeros((360,500))*np.nan
        for az in range(360):
            for r in range(500):
                advection_corrected_dbz[az][r] = data[az*500+r]
         
        advection_corrected_dbz[np.isnan(advection_corrected_dbz)] = -500
        single_sweep_data = current_to_be_corrected['sweep_0']
        vali = single_sweep_data['DBZH']
        root = current_to_be_corrected.copy()

        single_sweep_data['DBZH'].data = advection_corrected_dbz#, dims=single_sweep_data.dims, coords=single_sweep_data.coords)
        single_sweep_data['DBZH'].attrs['_Undetect'] = -500
        for var in ["CSP","VRADH","WRADH","ZDR","KDP","RHOHV","DBZHC","VRADDH","TH","SQIH","PHIDP","HCLASS","ZDRC","TV","SNR","PMI"]:
            single_sweep_data[var] = None
        
        root._children.clear()
        root._children["sweep_0"]= single_sweep_data
        
        xd.io.to_odim(root,'corrected_field/'+curtime_str+'.h5', source = "WIGOS:0-246-0-107275,RAD:FI53,PLC:Vihti,NOD:fivih")
        
        current_time += datetime.timedelta(minutes=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute ODIM h5 files using the advection correction mappings created in compute_advection_correction.py"
    )
    parser.add_argument("starttime", help="start time (YYYYmmddHHMM)")
    #parser.add_argument("config", help="configuration profile to use")
    args = parser.parse_args()
    starttime = datetime.strptime(args.starttime, "%Y%m%d%H%M")

    correct_radar_volumes(starttime)