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
import argparse

class advection_adjustment():
    def __init__(self,antenna_lat,antenna_lon,antenna_height):
        ground_level_dataset = xr.open_dataset("suomi_korkeusalueet_luvuilla.tif", decode_coords="all").isel(band=0).to_dataarray()
        #X,Y = np.meshgrid(,ground_level_dataset.y)
        self.ground_lons = ground_level_dataset.x.values
        self.ground_lats = ground_level_dataset.y.values
        #self.ground_level_cKDTree = cKDTree(np.vstack([Y.ravel(), X.ravel()]).T)
        self.ground_level_dataset = ground_level_dataset.values[0]
        self.weather = xr.open_dataset("smartmets/combined_data.nc", engine="netcdf4").to_dataarray()
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
    
    def get_nearest_xy_coordinate_in_model(self,x,y):
        dist, idx = self.cKDTree_weather.query((y, x))
        return np.unravel_index(idx, self.shape)
        
    def get_nearest_ground_level(self,x,y):
        i = np.argmin(np.abs(self.ground_lons - x))
        j = np.argmin(np.abs(self.ground_lats - y))
        return self.ground_level_dataset[j,i]

    def get_model_level(self,z):
        index = np.abs(self.heights - z)
        smallest_value = self.heights[np.argmin(index)]
        return self.height_to_level[smallest_value]-18
    
    def get_radar_bin_height(self, x, y, ground):
        lat_diff_m = (y-self.antenna_lat)
        lon_diff_m = (x-self.antenna_lon)*np.cos((np.deg2rad(y+self.antenna_lat)/2))
        
        distance_meters = np.sqrt(lat_diff_m**2+lon_diff_m**2)*111412
        if distance_meters > 250000:
            return np.nan
        
        return 6371000*4.0/3.0*(np.cos(np.deg2rad(0.3))/(np.cos(np.deg2rad(0.3)+distance_meters/(6371000*4.0/3.0)))-1)+ self.antenna_height - ground
    
    def rise_from_ml_given_time_from_x_0(self,time,x_0):
        # here time and x_0 are with respect to the beginning of melting layer
        return -np.exp(-time/175 + np.log(875-x_0))+875

    def time_in_the_melting_layer(self,x_0,x_l):
        # Note tämä on laskettu siten että x_0 on ml:n alakohta!
        return -175*np.log(875-x_l)+175*np.log(875-x_0)

    def compute_time_for_rise(self,rise,korkeus_alku,ml_height):
        timestep = 0
        aloitetaan_melting_layerin_paalta = korkeus_alku >= ml_height
        if aloitetaan_melting_layerin_paalta:
            return rise
        
        aloitetaan_melting_layerin_alta = korkeus_alku < ml_height-700
        
        if aloitetaan_melting_layerin_alta:
            nousu_jaa_melting_layerin_alle = rise+korkeus_alku < ml_height-700
            if nousu_jaa_melting_layerin_alle:
                return rise/5
            
            nousu_melting_layeriin = ((ml_height-700)-korkeus_alku)
            timestep += nousu_melting_layeriin/5
            rise -= nousu_melting_layeriin
            korkeus_alku = ml_height

        korkeus_alku_melting_layerin_pohjan_suhteen = korkeus_alku - (ml_height-700)
        korkeus_loppu_melting_layerin_pohjan_suhteen = korkeus_alku_melting_layerin_pohjan_suhteen + rise

        nousu_ylittaa_melting_layerin = korkeus_loppu_melting_layerin_pohjan_suhteen > 700
        if nousu_ylittaa_melting_layerin:
            timestep += korkeus_alku + rise - ml_height
            korkeus_loppu_melting_layerin_pohjan_suhteen = 700
        
        return timestep + self.time_in_the_melting_layer(korkeus_alku_melting_layerin_pohjan_suhteen, korkeus_loppu_melting_layerin_pohjan_suhteen)
        
    def compute_next_timestep(self, y, x, z, lvl, u_tuuli, v_tuuli, ml_height,closest_x,closest_y):
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

    def compute_rise(self, timestep, alku_korkeus, ml_height):
        nousun_kesto_melting_layeriin_jos_alku_korkeus_on_pienempi_kuin_ml_alaraja = max(ml_height-700-alku_korkeus,0)/5
        timestepissa_noustaan_melting_layeriin = nousun_kesto_melting_layeriin_jos_alku_korkeus_on_pienempi_kuin_ml_alaraja >= timestep
        
        if timestepissa_noustaan_melting_layeriin:
            return timestep*5
        
        nousu_alkaa_melting_layerin_paalta = alku_korkeus > ml_height
        if nousu_alkaa_melting_layerin_paalta:
            return timestep # sillä [aika]*1 m/s on matka

        nousu_melting_layeriin = nousun_kesto_melting_layeriin_jos_alku_korkeus_on_pienempi_kuin_ml_alaraja*5
        timestep_mahdollisen_melting_layeriin_nousun_jalkeen = timestep - nousun_kesto_melting_layeriin_jos_alku_korkeus_on_pienempi_kuin_ml_alaraja

        korkeus_melting_layerin_alarajasta = nousu_melting_layeriin+alku_korkeus-(ml_height-700)
        rise_from_ml_using_analytical_formula = self.rise_from_ml_given_time_from_x_0(timestep_mahdollisen_melting_layeriin_nousun_jalkeen,korkeus_melting_layerin_alarajasta)
        nousu_jaa_melting_layerin_sisaan = rise_from_ml_using_analytical_formula < 700

        if nousu_jaa_melting_layerin_sisaan:
            return (ml_height-700)+rise_from_ml_using_analytical_formula - alku_korkeus
        
        rise_in_ml = 700-(korkeus_melting_layerin_alarajasta)
        timestep_melting_layerin_ylaosassa = timestep_mahdollisen_melting_layeriin_nousun_jalkeen - self.time_in_the_melting_layer(korkeus_melting_layerin_alarajasta,700)
        
        return timestep_melting_layerin_ylaosassa + nousu_melting_layeriin + rise_in_ml
         
    def advection_from_a_grid_cell(self,lat,lon, current_time):
        ground = self.get_nearest_ground_level(lon,lat)        
        z = 0
        if np.isnan(ground): 
            ground = 0

        el_h = self.get_radar_bin_height(lon,lat,ground)
        
        time_correction = current_time.minute*60 + current_time.second
        t = 0
        lat_alku = lat
        lon_alku = lon
        closest_y, closest_x = self.get_nearest_xy_coordinate_in_model(lon,lat)
        
        first_timestamp = self.weather.time.data
        starting_hour = ((np.datetime64(current_time)-first_timestamp).astype('timedelta64[s]')/3600).astype(int)

        while el_h > z:
            step=int(np.floor((t+time_correction)/3600) + starting_hour)
            # tässä approksimaatio 1 degree lat on 111.412 km
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

            ground = self.get_nearest_ground_level(lon,lat)        
            if np.isnan(ground): 
                ground = 0

            lvl = self.get_model_level(z)
            u_tuuli = self.u_wind[step,lvl,closest_y,closest_x]
            v_tuuli = self.v_wind[step,lvl,closest_y,closest_x]
            ml_height = self.melting_layer_height[step,closest_y,closest_x]
            # tämä raskas
            timestep = self.compute_next_timestep(lat,lon,z,lvl,u_tuuli,v_tuuli,ml_height,closest_x,closest_y)
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

    def get_adjusted_dbz(self, starttime_str, reference_filename = "/arch/radar/HDF5/2022/07/07/radar/polar/fiuta/202207070000_radar.polar.fiuta.h5"
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
        
        start_time = datetime.datetime.strptime(starttime_str, "%Y%m%d%H%M")
        end_time = start_time+ datetime.timedelta(hours=3)
        current_time = start_time
        while current_time <= end_time:

            print("now computing:",current_time)
            def process(i,j):
                lat,lon = radar_lats[i,j],radar_lons[i,j]
                return self.advection_from_a_grid_cell(lat,lon, current_time)  

            #print((radar_lats))
            with parallel_backend("loky", inner_max_num_threads=2):
                data = Parallel(n_jobs=4,verbose=1)(delayed(process)(i,j,)  for i in range(0,360//step_azi) for j in range(0,500//step_r))
            
            time_vol = np.zeros((360//step_azi+1,500//step_r+1))*np.nan

            new_lat = np.zeros((360//step_azi+1,500//step_r+1))*np.nan
            new_lon = np.zeros((360//step_azi+1,500//step_r+1))*np.nan
            el_h = np.zeros((360//step_azi+1,500//step_r+1))*np.nan
            
            for i in range(0,360//step_azi):
                for j in range(0,500//step_r):
                    a = i*500//step_r+j
                    if data[a]:
                        if ~np.isnan(data[a][2]):
                            new_lat[i,-j-1] = data[a][0]
                            new_lon[i,-j-1] = data[a][1]
                            el_h[i,-j-1] = data[a][2]
                            time_vol[i,-j-1] = -data[a][3]/(30*5)

            for j in range(0,500//step_r):
                a = j
                if data[a]:
                    if ~np.isnan(data[a][2]):
                        new_lat[360//step_azi,-j-1] = data[a][0]
                        new_lon[360//step_azi,-j-1] = data[a][1]
                        el_h[360//step_azi,-j-1] = data[a][2]
                        time_vol[360//step_azi,-j-1]= -data[a][3]/(30*5)
            
            # täytetään toiselta puolelta ensimmäisellä tutkabinillä yleisesti nolla.
            for i in range(0,360//step_azi+1):
                u_i = (i +(180//step_azi)) % (360//step_azi)
                new_lat[i][0] = data[a][0]
                new_lon[i][0] = data[a][1]
                el_h[i][0] = data[a][2]
                time_vol[i][0] = time_vol[u_i][2]

            x = np.arange(0,361,step_azi)
            y = np.concatenate([np.array([-step_r]),np.array(np.arange(step_r,500 + step_r,step_r))])
            xg, yg = np.meshgrid(np.arange(0,360), np.arange(0,500))
            
            measured_points = (x,y)
            points_in_which_to_interpolate = np.array([xg.flatten(),yg.flatten()]).T
            
            interp_time = RegularGridInterpolator(measured_points, time_vol)     
            final_time = interp_time(points_in_which_to_interpolate).reshape((500,360)).T.round()

            interp_lat = RegularGridInterpolator(measured_points, new_lat)     
            final_lat = interp_lat(points_in_which_to_interpolate).reshape((500,360)).T

            interp_lon = RegularGridInterpolator(measured_points, new_lon)     
            final_lon = interp_lon(points_in_which_to_interpolate).reshape((500,360)).T

            interp_el_h = RegularGridInterpolator(measured_points, el_h)     
            final_el_h = interp_el_h(points_in_which_to_interpolate).reshape((500,360)).T

            cur_str = datetime.datetime.strftime(current_time, "%Y%m%d%H%M")
            np.save('correction_maps/'+cur_str+'_lon', final_lon)
            np.save('correction_maps/'+cur_str+'_lat', final_lat)
            np.save('correction_maps/'+cur_str+'_time', final_time)
            np.save('correction_maps/'+cur_str+'_el', final_el_h)

            current_time += datetime.timedelta(minutes=5)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute ODIM h5 files using the advection correction mappings created in compute_advection_correction.py"
    )
    parser.add_argument("starttime", help="start time (YYYYmmddHHMM)")
    #parser.add_argument("config", help="configuration profile to use")
    args = parser.parse_args()
    starttime = args.starttime
    
    advec = advection_adjustment(64.7749301232398,26.3188800774515,118)
    # hae kellonaika ja tee edellisen tunnin perusteella tuo homma.

    data = advec.get_adjusted_dbz(starttime)