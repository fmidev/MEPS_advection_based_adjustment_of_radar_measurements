# alimmat kulmat PPI1_A, PPI2_A ja PPI3_A:n  matala kulma (0.3) näkee kauas (Luostolla 0.1).
# 250 km etäisyydellä maanpinnalla ollaan korkeudessa noin 5 km

# alin smartmet on 12 m
# korkeus_malli_mappi

# Olisi varmaan hyvä olla height to level sittenkin.
import xarray as xr
import numpy as np
import wradlib as wrl
import xradar as xd
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import datetime


class advection_adjustment():
    def __init__(self,antenna_lat,antenna_lon,antenna_height):
        ground_level_dataset = xr.open_dataset("suomi_korkeusalueet_luvuilla.tif", decode_coords="all").isel(band=0).to_dataarray()
        #X,Y = np.meshgrid(,ground_level_dataset.y)
        self.ground_lons = ground_level_dataset.x.values
        self.ground_lats = ground_level_dataset.y.values
        #self.ground_level_cKDTree = cKDTree(np.vstack([Y.ravel(), X.ravel()]).T)
        self.ground_level_dataset = ground_level_dataset.values[0]
        self.weather = xr.open_dataset("smartmets/combined_data.nc", decode_coords="all").to_dataarray()
        self.u_wind = self.weather.sel(variable='u').data
        self.v_wind = self.weather.sel(variable='v').data
        self.height_to_level = {7573.1:18,7156.3:19,6757.3:20,6375.0: 21,6008.5: 22,5657.1: 23,5320.3: 24, 4997.4: 25, 4688.0: 26, 4391.7: 27, 4108.1: 28, 3836.9: 29, 3577.8: 30, 3330.4: 31, 3094.5: 32, 2870.0: 33, 2656.5: 34, 2454.1: 35, 2262.5: 36, 2081.5: 37, 1911.2: 38, 1751.4: 39, 1602.1: 40, 1463.1: 41, 1334.4: 42, 1215.0: 43, 1104.4: 44, 1001.8: 45, 906.8: 46, 819.0: 47, 737.8: 48, 662.9: 49, 593.8: 50, 530.1: 51, 471.5: 52, 417.6: 53, 368.1: 54, 322.6: 55, 280.7: 56, 242.2: 57, 206.6: 58, 173.7: 59, 143.1: 60, 114.5: 61, 87.5: 62, 61.7: 63, 36.8: 64, 12.2: 65}
        self.level_to_height = {18:7573.1,19:7156.3,20:6757.3,21:6375.0,22:6008.5,23:5657.1,24: 5320.3, 25: 4997.4, 26: 4688.0, 27: 4391.7, 28: 4108.1, 29: 3836.9, 30: 3577.8, 31: 3330.4, 32: 3094.5, 33: 2870.0, 34: 2656.5, 35: 2454.1, 36: 2262.5, 37: 2081.5, 38: 1911.2, 39: 1751.4, 40: 1602.1, 41: 1463.1, 42: 1334.4, 43: 1215.0, 44: 1104.4, 45: 1001.8, 46: 906.8, 47: 819.0, 48: 737.8, 49: 662.9, 50: 593.8, 51: 530.1, 52: 471.5, 53: 417.6, 54: 368.1, 55: 322.6, 56: 280.7, 57: 242.2, 58: 206.6, 59: 173.7, 60: 143.1, 61: 114.5, 62: 87.5, 63: 61.7, 64: 36.8, 65: 12.2}
        self.shape = self.weather['latitude'].values.shape
        melting_layer_height= (np.abs(self.weather.sel(variable='t')-273.15).argmin(dim='hybrid'))+18
        melting_layer_height.data = np.vectorize(self.level_to_height.get)(melting_layer_height)
        self.melting_layer_height=melting_layer_height.data
        self.heights = np.array(list(self.height_to_level.keys()))
        self.lats = self.weather['latitude'].values
        self.lons = self.weather['longitude'].values
        self.cKDTree_weather = cKDTree(np.column_stack((self.weather['latitude'].values.flatten(), self.weather['longitude'].values.flatten())))
        self.antenna_lat = antenna_lat
        self.antenna_lon = antenna_lon
        self.antenna_height = antenna_height
    
    def get_closest_xy_coordinate_in_model(self,x,y):
        dist, idx = self.cKDTree_weather.query((y, x))
        return np.unravel_index(idx, self.shape)
        
    def get_closest_ground_level(self,x,y):
        #out = float(self.ground_level_dataset.sel(y=y,x=x,method="nearest").data)
        #_, idx = self.ground_level_cKDTree.query((y, x))
        i = np.argmin(np.abs(self.ground_lons - x))
        j = np.argmin(np.abs(self.ground_lats - y))
        #print(j,i,flush=True)
        return self.ground_level_dataset[j,i]

    # nearest value selection
    def get_model_level(self,z):
        index = np.abs(self.heights - z)
        smallest_value = self.heights[np.argmin(index)]
        return self.height_to_level[smallest_value]-18
    
    def get_radar_bin_height(self, x, y, ground):
        #distance_meters=geopy.distance.geodesic((y, x), (self.antenna_lat, self.antenna_lon)).meters
        lat_diff_m = (y-self.antenna_lat)
        lon_diff_m = (x-self.antenna_lon)*np.cos((np.deg2rad(y+self.antenna_lat)/2))
        #approximate_distance
        distance_meters = np.sqrt(lat_diff_m**2+lon_diff_m**2)*111412
        # doviak 2.28c, johon lisätty antennin korkeus
        #print(distance_meters)
        if distance_meters > 250000:
            return np.nan
        return 6371000*4.0/3.0*(np.cos(np.deg2rad(0.3))/(np.cos(np.deg2rad(0.3)+distance_meters/(6371000*4.0/3.0)))-1)+ self.antenna_height - ground

    def rise_from_ml_given_time_from_x_0(self,time,x_0):
        # t = int_{x(0)}^{x(time)}1/(5-4x/700)dx, x(0)=0
        # from here https://www.wolframalpha.com/input?i=integral+of+1%2F%285-4x%2F700%29
        # x(time)=?
        # time = -175*np.log(875-x(time))+175*np.log(875-x(0))
        # -time/175 = np.log(875-x(time))-np.log(875-x_0)
        # -time/175 + np.log(875-x_0) = np.log(875-x(time)) | eksponentti funktio
        # np.exp(-time/175 + np.log(875-x_0)) = 875-x(time)
        # np.exp(-time/175 + np.log(875-x_0))-875 = -x(time)
        # -np.exp(-time/175 + np.log(875-x_0))+875 = x(time)
        # here time and x_0 are with respect to the beginning of melting layer
        return -np.exp(-time/175 + np.log(875-x_0))+875
    
    def time_in_the_melting_layer(self,x_0,x_l):
        # Note tämä on laskettu siten että x_0 on ml:n alakohta!
        return -175*np.log(875-x_l)+175*np.log(875-x_0)

    def compute_time_for_rise(self,rise,z,ml_height):
        if rise+z < ml_height-700:
            return rise/5
        elif z >= ml_height:
            return rise
        else:
            # Väistämättä käydään ml:ssä
            timestep = self.time_in_the_melting_layer(max(0,z-(ml_height-700)),min(max(0,z+rise-(ml_height-700)),700))
            
            # vielä on mahdollista, että 
            if z+rise >= ml_height:
                #matka ml_stäylös/1
                timestep += max(0, z+rise-ml_height)

            if z < ml_height - 700:
                timestep += max(0,(ml_height-700)-z)/5

            return timestep
    
    def compute_next_timestep(self, y, x, z, lvl, u_tuuli, v_tuuli, ml_height, t, step,closest_x,closest_y):
        timestep = 100000 # Sekunti
        
        if u_tuuli > 0 and closest_x < self.shape[1]:
            lat = self.lats[closest_y,closest_x + 1]
            lon = self.lons[closest_y,closest_x + 1]
            lat_diff_m = (y-lat)
            lon_diff_m = (x-lon)*np.cos((np.deg2rad(y+lat)/2))
            dist = np.sqrt(lat_diff_m**2+lon_diff_m**2)*111412
            timestep = min(dist/np.abs(u_tuuli), timestep)
        elif u_tuuli < 0 and closest_x > 0:
            lat = self.lats[closest_y,closest_x - 1]
            lon = self.lons[closest_y,closest_x - 1]
            lat_diff_m = (y-lat)
            lon_diff_m = (x-lon)*np.cos((np.deg2rad(y+lat)/2))
            dist = np.sqrt(lat_diff_m**2+lon_diff_m**2)*111412
            timestep = min(dist/np.abs(u_tuuli), timestep)
        
        if v_tuuli > 0 and closest_y < self.shape[0]:
            lat = self.lats[closest_y + 1,closest_x]
            lon = self.lons[closest_y + 1,closest_x]
            lat_diff_m = (y-lat)
            lon_diff_m = (x-lon)*np.cos((np.deg2rad(y+lat)/2))
            dist = np.sqrt(lat_diff_m**2+lon_diff_m**2)*111412
            timestep = min(dist/np.abs(v_tuuli), timestep)
        elif v_tuuli < 0 and closest_y > 0:
            lat = self.lats[closest_y - 1,closest_x]
            lon = self.lons[closest_y - 1,closest_x]
            lat_diff_m = (y-lat)
            lon_diff_m = (x-lon)*np.cos((np.deg2rad(y+lat)/2))
            dist = np.sqrt(lat_diff_m**2+lon_diff_m**2)*111412
            timestep = min(dist/np.abs(v_tuuli), timestep)        
        
        to_next_height = max(0,self.level_to_height[lvl+17]-self.level_to_height[lvl+18])
        timestep = min(self.compute_time_for_rise(to_next_height,z,ml_height),timestep)
        
        return max(1, timestep)

    def compute_rise(self, timestep, z, ml_height):
        # 1. nousun kesto ml_heightiin -700
        kesto = max(ml_height-700-z,0)/5
        # jos kesto on suurempi kuin timestep voidaan palauttaa arvo
        
        if kesto >= timestep:
            return timestep*5
        elif z > ml_height:
            return timestep
        else:
            rise = kesto*5
            timestep -= kesto
            x_0 = rise+z
            rise_from_ml = self.rise_from_ml_given_time_from_x_0(timestep, x_0-(ml_height-700))
            if rise_from_ml < 700:
                return (ml_height-700)+rise_from_ml - z
            else:
                rise_in_ml = 700-(x_0-(ml_height-700))
                timestep -= self.time_in_the_melting_layer(x_0-(ml_height-700),700)
                return timestep + rise + rise_in_ml

    def advection_from_a_grid_cell(self,lat,lon):             
        
        ground = self.get_closest_ground_level(lon,lat)        
        z = 0
        if np.isnan(ground): 
            ground = 0

        el_h = self.get_radar_bin_height(lon,lat)
        #Yksi iteraation on sekunti
        t = 0
        lat_alku = lat
        lon_alku = lon
        #start_time = time.time()
        closest_y, closest_x = self.get_closest_xy_coordinate_in_model(lon,lat)

        while el_h > z:
            # Naivi aikakehitys            
            step=-(int(-t)//3600) 
            # Katso ollaanko liikuttu seuraavaan ruutuun
            # tässä approksimaatio 1 degree lat on 111.412 km
            # jos heittää lat lon veks. voidaan zxy laskea kerralla...
            if np.abs(lat-lat_alku) > 2.5/111.412:
                if lat > lat_alku:
                    closest_y +=1
                else:
                    closest_y -=1
                lat_alku = lat
            # tässä approksimaatio 1 degree lon on 111.412 km *np.cos(np.deg2rad(lat))
            if np.abs(lon-lon_alku) > 2.5/(111.412*np.cos(np.deg2rad(lat))):
                if lon > lon_alku:
                    closest_x +=1
                else:
                    closest_x -=1
                lon_alku = lon

            # Paljonko on maantaso
            ground = self.get_closest_ground_level(lon,lat)        
            if np.isnan(ground): 
                ground = 0

            lvl = self.get_model_level(z)
            u_tuuli = self.u_wind[step,lvl,closest_y,closest_x]
            v_tuuli = self.v_wind[step,lvl,closest_y,closest_x]
            ml_height = self.melting_layer_height[step,closest_y,closest_x]
            # tämä raskas
            timestep = self.compute_next_timestep(lat,lon,z,lvl,u_tuuli,v_tuuli,ml_height,t,step,closest_x,closest_y)
            lon -= u_tuuli*timestep/(111412*np.cos(np.deg2rad(lat)))
            lat -= v_tuuli*timestep/111412
            z += self.compute_rise(timestep,z,ml_height)
            t -= timestep

            el_h = self.get_radar_bin_height(lon,lat, ground)
            
        # approximation to correct the last movement
        to_next_height = max(0,z-el_h)
        timestep = self.compute_time_for_rise(to_next_height,el_h,ml_height)
        
        # Tässä lisätään timestep, jotta saadaan hetki jolloin alin kulma leikkaa.
        return (lat,lon,el_h,t+timestep)

    def get_adjusted_dbz(self, start_time, end_time,reference_filename = "/home/myllykos/Documents/mepsi_testisetti/202402132255_fivih_PVOL.h5"
        ):
        pvol = xd.io.open_odim_datatree(reference_filename)

        ds1 = pvol["sweep_0"].ds.wrl.georef.georeference(
            crs=wrl.georef.get_default_projection()
        )
        step_azi = 5
        step_r = 5
        #Näissä viimeinen jää pois lasketaan ne erikseen.
        radar_lats = ds1.to_dataarray().y.values[::step_azi,::-step_r]
        radar_lons = ds1.to_dataarray().x.values[::step_azi,::-step_r]
        print(radar_lats)

        def process(i,j):
            lat,lon = radar_lats[i,j],radar_lons[i,j]
            return self.advection_from_a_grid_cell(lat,lon)  

        print((radar_lats))
        start_time = time.time()
        with parallel_backend("loky", inner_max_num_threads=2):
            data = Parallel(n_jobs=4,verbose=1)(delayed(process)(i,j)  for i in range(0,360//step_azi) for j in range(0,500//step_r))
        print(time.time()-start_time) 

        # Tästä eteenpäin uusiksi.

        #print(len(data))
        #print(data[0].shape)
        vali_time = time.time()
        time_vol = np.zeros((360//step_azi+1,500//step_r+1))*np.nan

        new_lat = np.zeros((360//step_azi+1,500//step_r+1))*np.nan
        new_lon = np.zeros((360//step_azi+1,500//step_r+1))*np.nan
        el_h = np.zeros((360//step_azi+1,500//step_r+1))*np.nan
        
        for i in range(0,360//step_azi):
            for j in range(0,500//step_r):
                    a = i*500//step_r+j
                    if data[a]:
                        # jakaja liittyy aikaan
                        if ~np.isnan(data[a][2]):
                            new_lat[i,-j-1] = data[a][0]
                            new_lon[i,-j-1] = data[a][1]
                            el_h[i,-j-1] = data[a][2]
                            time_vol[i,-j-1] = -data[a][3]//(30*5)

        for j in range(0,500//step_r):
            a = j
            if data[a]:
                if ~np.isnan(data[a][2]):
                    # jakaja liittyy aikaan
                    new_lat[360//step_azi,-j-1] = data[a][0]
                    new_lon[360//step_azi,-j-1] = data[a][1]
                    el_h[360//step_azi,-j-1] = data[a][2]
                    time_vol[360//step_azi,-j-1]= -data[a][3]//(30*5)
        
        # täytetään toiselta puolelta ensimmäisellä tutkabinillä yleisesti nolla.
        for i in range(0,360//step_azi+1):
            u_i = (i +(180//step_azi)) % (360//step_azi)
            new_lat[i][0] = data[a][0]
            new_lon[i][0] = data[a][1]
            el_h[i][0] = data[a][2]
            time_vol[i][0] = time_vol[u_i][2]

        x = np.arange(0,361,step_azi)
        # Tässä tehdään näin koska nolla bini puuttuu laskuista ja toisen puolen ensimmäinen tutkabini vastaa indeksiä -step
        y = np.concatenate([np.array([-step_r]),np.array(np.arange(step_r,500 + step_r,step_r))])
        xg, yg = np.meshgrid(np.arange(0,360), np.arange(0,500))

        interp_time = RegularGridInterpolator((x,y), time_vol)     
        final_time = interp_time(np.array([xg.flatten(),yg.flatten()]).T).reshape((500,360)).T.round()
        
        ds1["# of radar volumes to past for advection correction"] = (['azimuth','range'], final_time)
        ds1["# of radar volumes to past for advection correction"].plot(x="x", y="y", cmap="viridis")
        
        plt.show()
        plt.close()
        final_time =  final_time.astype(int)

        interp_lat = RegularGridInterpolator((x,y), new_lat)     
        final_lat = interp_lat(np.array([xg.flatten(),yg.flatten()]).T).reshape((500,360)).T

        interp_lon = RegularGridInterpolator((x,y), new_lon)     
        final_lon = interp_lon(np.array([xg.flatten(),yg.flatten()]).T).reshape((500,360)).T

        interp_el_h = RegularGridInterpolator((x,y), el_h)     
        final_el_h = interp_el_h(np.array([xg.flatten(),yg.flatten()]).T).reshape((500,360)).T
        
        ds1["# of radar volumes to past for advection correction"] = (['azimuth','range'], final_time)
        #ds1["height of the bin"] = (['azimuth','range'], final_el_h)
        ds1["mapped latitude"] = (['azimuth','range'], final_lat)
        ds1["mapped longitude"] = (['azimuth','range'], final_lon)
        vali_time= time.time()
        
        """
        advection_corrected_dbz = np.zeros((360,500))*np.nan
        # Tee lista, jossa on avatut tiedostot timestamppeinä menneeseen.
        # tässä periaatteessa riittää kerta.
        moments = {}
        timestamp = "202402132255"
        dt = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M")

        for t_i in range(final_time[~np.isnan(final_time)].max()+1):
            filename = "/home/myllykos/Documents/mepsi_testisetti/"+timestamp+"_fivih_PVOL.h5"
            dt-=datetime.timedelta(minutes=5)
            timestamp = datetime.datetime.strftime(dt,"%Y%m%d%H%M")
            pvol = xd.io.open_odim_datatree(filename)

            moments[t_i] = pvol["sweep_0"].ds.wrl.georef.georeference(
                crs=wrl.georef.get_default_projection()
            )['DBZH'].data

            print("valitime3",time.time()-vali_time)
        
        vali = ds1.to_dataarray()
        cKDTree_radar = cKDTree(np.column_stack((vali.y.values.flatten(), vali.x.values.flatten())))
        #   keksi keino, jolla saadaan lat, lon parista lähin tutkan az,r pari.
        
        def get_closest_radar_coordinate(x,y):
            dist, idx = cKDTree_radar.query((y, x))
            return np.unravel_index(idx, (360,500))
        print("valitime3.5",time.time()-vali_time)
        # Tämän voi rinnakaistaa...
        
        def collect_correction(az,r):
            if ~np.isnan(final_el_h[az,r]):
                # kerää tästä lähin ajanhetki.
                az_radar,r_radar = get_closest_radar_coordinate(final_lon[az,r],final_lat[az,r])
                #print(az_radar,r_radar)
                return moments[final_time[az,r]][az_radar,r_radar]
            return np.nan

        with parallel_backend("loky", inner_max_num_threads=2):
            data = Parallel(n_jobs=4,verbose=1)(delayed(collect_correction)(i,j)  for i in range(0,360) for j in range(0,500))
         
        for az in range(360):
            for r in range(500):
                advection_corrected_dbz[az][r] = data[az*500+r]

        ds1["advection_corrected_DBZH"] = (['azimuth','range'], advection_corrected_dbz)
        ds1["advection_corrected_DBZH"].plot(x="x", y="y", cmap="viridis",vmin=-10, vmax=50)
        print("valitime4",time.time()-vali_time)
        
        plt.show()
        plt.close()
        ds1["DBZH"].plot(x="x", y="y", cmap="viridis",vmin=-10, vmax=50)
        
        plt.show()
        plt.close()
        #xradar.io.export.odim.to_odim(dtree, filename
        return ds1
        """
        return None
    
st = time.time()
advec = advection_adjustment(60.5561900138855,24.4955920055509,181)
		
import pickle
data = advec.get_adjusted_dbz()
print("time outside",time.time()-st) 
#with open('last_output.pickle', 'wb') as f:
#    pickle.dump(data, f)

#xr.merge([temperature_ds, pressure_ds, humidity_ds])
#ax1, dem = wrl.vis.plot_ppi(
#    polarvalues, ax=ax1, r=r, az=coord[:, 0, 1], cmap=mpl.cm.terrain, vmin=0.0
#)
# nykyhetkestä ja mennä menneeseen koska menneisyydestä tulevaisuuteen voidaan joutua mappaamaan samalle arvolle, 
# kun taas toisin päin ei tule ongelmaa sillä aina on menneisyydessä arvo...
"""
def correct_volume_based_on_lowest_elevation():
    # aloita tämän hetken volyymistä ja tee siitä eteenpäin, sillä silloin on jo suora korkeus.
    # Ekalle tunnille erotus kertaa tuuli suunnan asetus ja sitten lopuille tehdään niin, että laitetaan oikeisiin suuntiin asiat yksinkertaisemmilla kertoimilla.
    # Laske alimmat kohdat jokaiselle alueen sijainnille niille kohdille, jotka tutka havaitsee.
    
    # tehdään vaakasuunnassa lineaarinen interpolointi

"""