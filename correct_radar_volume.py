import numpy as np
from scipy.spatial import cKDTree
import datetime
from joblib import Parallel, delayed, parallel_backend
import wradlib as wrl
import xradar as xd
import os
import glob


def correct_radar_volumes(starttime_str):
    # Tee lista, jossa on avatut tiedostot timestamppeinä menneeseen.
    # tässä periaatteessa riittää kerta.
    moments = {}
    datet_orig = datetime.datetime.strptime(starttime_str, "%Y%m%d%H%M")
    datet = datet_orig
    first_times = np.load("correction_maps/"+starttime_str+"_time.npy")
    
    # Avaa muistiin alkuun kaikki noi oleelliset, sitten seuraavat ja sitten seuraavat etc.
    # laske korjaus yhdelle
    # sitten hae seuraavan tarpeet ja hyödynnä vanhoja hyödyksi
    # laske korjaus yhdelle
    # jne...
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
    #   keksi keino, jolla saadaan lat, lon parista lähin tutkan az,r pari.
    
    def get_closest_radar_coordinate(x,y):
        dist, idx = cKDTree_radar.query((y, x))
        return np.unravel_index(idx, (360,500))
    

    # tähän looppi ajan yli.
    # 1. avaa filut. 
    # 2. loop min max yli ja katso tarvitseeko uutta dt lisätä momentsiin.
    # 3. aja sitten collect collectionit.
    # 4. tallenna data esim. kopioimalla data alkuperäisen volyymin kopioon 

    end_time = datet_orig + datetime.timedelta(hours=3)
    current_time = datet_orig
    while current_time <= end_time:
        curtime_str = current_time.strftime("%Y%m%d%H%M")
        print("Now correcting:",curtime_str) 
        final_lat = np.load("correction_maps/"+curtime_str+"_lat.npy")
        final_lon = np.load("correction_maps/"+curtime_str+"_lon.npy")
        final_times = np.load("correction_maps/"+curtime_str+"_time.npy")

        last_in_past = int(final_times[~np.isnan(final_times)].max())
        # loop until the last in past to add the necessary values to the moments dictionary
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
        """
        ds1["DBZH"].plot(x="x", y="y", cmap="viridis",vmin=-10, vmax=50)
        
        #xradar.io.export.odim.to_odim(dtree, filename
        return ds1
        """
        #print(pvol)
        single_sweep_data = pvol['sweep_0']
        cleaned_data = single_sweep_data.drop_vars([var for var in single_sweep_data.data_vars if var != 'DBZH'])
        cleaned_data['advection_corrected'] = advection_corrected_dbz
        xd.io.to_odim(cleaned_data,'corrected_field/'+curtime_str+'.h5', source = ???)
        break
        current_time += datetime.timedelta(minutes=5)

if __name__ == '__main__':
    
    timestr = "202411080600"
    #           202411070700
    correct_radar_volumes(timestr)
    # poista vanhat
    #for file in glob.glob(os.path.join("correction_maps", "*.npy")):
    #    os.remove(file)