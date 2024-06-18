# alimmat kulmat PPI1_A, PPI2_A ja PPI3_A:n  matala kulma (0.3) näkee kauas (Luostolla 0.1).
# 250 km etäisyydellä maanpinnalla ollaan korkeudessa noin 5 km

# alin smartmet on 12 m
# korkeus_malli_mappi

# Olisi varmaan hyvä olla height to level sittenkin.
import xarray as xr
import rioxarray
import numpy as np
import wradlib as wrl
import geopy.distance
import xradar as xd
import time


K_EA=6371000*4.0/3.0
EL = np.deg2rad(0.3)

class advection_adjustment():
    def __init__(self,antenna_lat,antenna_lon,antenna_height):
        self.ground_level_dataset = xr.open_dataset("suomi_korkeusalueet_luvuilla.tif", decode_coords="all").to_dataarray()
        self.weather = xr.open_dataset("latest_smartmet.grib", engine="cfgrib", decode_coords="all").to_dataarray()
        self.u_wind = self.weather.sel(variable='u').data
        self.v_wind = self.weather.sel(variable='v').data
        self.height_to_level = {5320.3: 24, 4997.4: 25, 4688.0: 26, 4391.7: 27, 4108.1: 28, 3836.9: 29, 3577.8: 30, 3330.4: 31, 3094.5: 32, 2870.0: 33, 2656.5: 34, 2454.1: 35, 2262.5: 36, 2081.5: 37, 1911.2: 38, 1751.4: 39, 1602.1: 40, 1463.1: 41, 1334.4: 42, 1215.0: 43, 1104.4: 44, 1001.8: 45, 906.8: 46, 819.0: 47, 737.8: 48, 662.9: 49, 593.8: 50, 530.1: 51, 471.5: 52, 417.6: 53, 368.1: 54, 322.6: 55, 280.7: 56, 242.2: 57, 206.6: 58, 173.7: 59, 143.1: 60, 114.5: 61, 87.5: 62, 61.7: 63, 36.8: 64, 12.2: 65}
        self.level_to_height = {24: 5320.3, 25: 4997.4, 26: 4688.0, 27: 4391.7, 28: 4108.1, 29: 3836.9, 30: 3577.8, 31: 3330.4, 32: 3094.5, 33: 2870.0, 34: 2656.5, 35: 2454.1, 36: 2262.5, 37: 2081.5, 38: 1911.2, 39: 1751.4, 40: 1602.1, 41: 1463.1, 42: 1334.4, 43: 1215.0, 44: 1104.4, 45: 1001.8, 46: 906.8, 47: 819.0, 48: 737.8, 49: 662.9, 50: 593.8, 51: 530.1, 52: 471.5, 53: 417.6, 54: 368.1, 55: 322.6, 56: 280.7, 57: 242.2, 58: 206.6, 59: 173.7, 60: 143.1, 61: 114.5, 62: 87.5, 63: 61.7, 64: 36.8, 65: 12.2}
        self.shape = self.weather['latitude'].values.shape
        melting_layer_height= (np.abs(self.weather.sel(variable='t')-273.15).argmin(dim='hybrid'))+24
        melting_layer_height.data = np.vectorize(self.level_to_height.get)(melting_layer_height)
        self.melting_layer_height=melting_layer_height.data
        self.heights  = np.array(list(self.height_to_level.keys()))
        self.lons = self.weather['longitude'].values
        self.lats = self.weather['latitude'].values
        self.flat_lons = self.weather['longitude'].values.flatten()
        self.flat_lats = self.weather['latitude'].values.flatten()
        self.antenna_lat = antenna_lat
        self.antenna_lon = antenna_lon
        self.antenna_height = antenna_height
    
    def get_closest_xy_coordinate_in_model(self,x,y):
        closest = np.sqrt((self.flat_lats - y)**2 + (self.flat_lons - x)**2).argmin()
        return np.unravel_index(closest, self.shape)

    # nearest value selection
    def get_model_level(self,z):
        index = np.abs(self.heights - z)
        smallest_value = self.heights[np.argmin(index)]
        return self.height_to_level[smallest_value]-24
    
    def get_radar_bin_height(self, x, y):
        distance_meters=geopy.distance.geodesic((y, x), (self.antenna_lat, self.antenna_lon)).meters
        # doviak 2.28c, johon lisätty antennin korkeus
        return K_EA*(np.cos(EL)/(np.cos(EL+distance_meters/(K_EA)))-1)+ self.antenna_height

    def get_fall_speed(self,z,melting_layer):
        return 5 - min(4, (4/700)*max(z-(melting_layer-700),0)) #m/s

    def compute_new_lon_lat_to_past(self,x,delta_x,y,delta_y):
        #Bearing in degrees: 0 – North, 90 – East, 180 – South, 270 or -90 – West.
        bearing_y = 180 if delta_y > 0 else 0
        bearing_x = -90 if delta_x > 0 else 90
        x = geopy.distance.distance(meters=delta_x).destination((y, x), bearing=bearing_x).longitude
        y = geopy.distance.distance(meters=delta_y).destination((y, x), bearing=bearing_y).latitude
        return (x,y)

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
        # Kolme osaa!
        # paljonko noususta on 1 m/s
        speed_of_start = self.get_fall_speed(z, ml_height)
        speed_of_risen = self.get_fall_speed(rise+z, ml_height)
        if speed_of_risen == 5:
            return rise/5
        elif speed_of_start == 1:
            return rise
        else:
            # Väistämättä käydään ml:ssä
            timestep = self.time_in_the_melting_layer(max(0,z-(ml_height-700)),min(max(0,z+rise-(ml_height-700)),700))
            
            # vielä on mahdollista, että 
            if speed_of_risen == 1:
                #matka ml_stäylös/1
                timestep += max(0, z+rise-ml_height)

            if speed_of_start ==5:
                timestep += max(0,(ml_height-700)-z)/5

            return timestep
    
    def compute_next_timestep(self, y, x, z, lvl, u_tuuli, v_tuuli, el_h, ml_height, t, step):
        #1. Mikä on etäisyys ajassa saman lvl:n ruudukkoon vallitsevan tuulen suunnan mukaisesti?
        # tuulen suunta, closest_y ja closest_x viereinen lokero, siitä arvo ja menoksi tai reunoilla 1250
        timestep = 100000 # Sekunti
        closest_y, closest_x = self.get_closest_xy_coordinate_in_model(x,y)
        
        if u_tuuli > 0 and closest_x < self.shape[1]:
            lat = self.lats[closest_y,closest_x + 1]
            lon = self.lons[closest_y,closest_x + 1]
            timestep = min(geopy.distance.geodesic((y, x), (lat, lon)).meters/np.abs(u_tuuli), timestep)
        elif u_tuuli < 0 and closest_x > 0:
            lat = self.lats[closest_y,closest_x - 1]
            lon = self.lons[closest_y,closest_x - 1]
            timestep = min(geopy.distance.geodesic((y, x), (lat, lon)).meters/np.abs(u_tuuli), timestep)

        if v_tuuli > 0 and closest_y < self.shape[0]:
            lat = self.lats[closest_y + 1,closest_x]
            lon = self.lons[closest_y + 1,closest_x]
            timestep = min(geopy.distance.geodesic((y, x), (lat, lon)).meters/np.abs(v_tuuli), timestep)
        elif v_tuuli < 0 and closest_y > 0:
            lat = self.lats[closest_y - 1,closest_x]
            lon = self.lons[closest_y - 1,closest_x]
            timestep = min(geopy.distance.geodesic((y, x), (lat, lon)).meters/np.abs(v_tuuli), timestep)
        #print("1",timestep)
        #2. Mikä on etäisyys ajassa ylempään ruudukkoon
        # Oleellisesti, puoliväli nykylvl:n ja seuraavan välissä. laske siis etäisyys z ja pyöristä ylös itse raja on aina ylempää palikkaa
        if lvl > 0:
            to_next_height = max(0,self.level_to_height[lvl+23]-self.level_to_height[lvl+24])
            timestep = min(self.compute_time_for_rise(to_next_height,z,ml_height),timestep)
            #print("2",timestep)
        #3. Mikä on etäisyys ajassa bin heightiin tai ainakin lähelle + 1 metri?
        # bin heightin ja z:n puoliväli + sekuntin nousu siitä, ja laske kuluva aika
        to_bin_height = max(0,el_h-z)
        timestep = min(self.compute_time_for_rise(to_bin_height,z,ml_height),timestep)
        #print("3",timestep)
        #4. Mikä on etäisyys seuraavaan mallisteppiin
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
        z = self.ground_level_dataset.sel(y=lat,x=lon,method="nearest").data[0,0]
        if np.isnan(z):
            return None
        el_h = self.get_radar_bin_height(lon,lat)
        #Yksi iteraation on sekunti
        t = 0
        step=0
        while el_h > z:
            #start_time = time.time()
            closest_y, closest_x = self.get_closest_xy_coordinate_in_model(lon,lat)
            lvl = self.get_model_level(z)
            
            u_tuuli = self.u_wind[step,lvl,closest_y,closest_x]
            v_tuuli = self.v_wind[step,lvl,closest_y,closest_x]
            ml_height = self.melting_layer_height[step,closest_y,closest_x]

            timestep = self.compute_next_timestep(lat,lon,z,lvl,u_tuuli,v_tuuli,el_h,ml_height,t,step)
            
            wind_from_west_to_east_in_timestep = u_tuuli*timestep
            wind_from_south_to_north_in_timestep = v_tuuli*timestep
            
            lon,lat = self.compute_new_lon_lat_to_past(lon,wind_from_west_to_east_in_timestep,lat,wind_from_south_to_north_in_timestep)

            # risessä jokin pahasti pielessä
            rise = self.compute_rise(timestep,z,ml_height)
            
            if rise <= 0:
                print("rise on negatiivinen VIRHE",rise)
                print(ml_height, z, timestep)
                print("before",z,"after", z+rise)
            z += rise
            t -= timestep
            el_h = self.get_radar_bin_height(lon,lat)
            
        return (lat,lon,el_h,t,z)

    def get_adjusted_dbz(self):
        filename = "202208281555_fianj_PVOL.h5"
        pvol = xd.io.open_odim_datatree(filename)

        ds1 = pvol["sweep_0"].ds.wrl.georef.georeference(
            crs=wrl.georef.get_default_projection()
        )
        
        for i, az in enumerate(ds1.azimuth.data):
            for j,ra in enumerate(ds1.range.data):
                start_time = time.time()
        
                location_values = ds1.sel(azimuth=az,range=ra)
                lat = location_values.y.data
                lon = location_values.x.data
                self.advection_from_a_grid_cell(lat,lon)  
             
            print(j,time.time()-start_time)
        
        # ds1 sisältää azimuth ja range pareina x,y coordsit.
        # Loopataan siis azimuth, range pareina data 
        # mapataan ne x ja y coordeille ja mennään näillä eteen päi
        # ja saadaan siten oleellinen kama ulos
        #print(ds1.coords)

advec = advection_adjustment(60.9038700163364,27.1080600656569,139)
print(advec.get_adjusted_dbz())
      #.advection_from_a_grid_cell(26.62972752573382,60.869673308673676))

# nykyhetkestä ja mennä menneeseen koska menneisyydestä tulevaisuuteen voidaan joutua mappaamaan samalle arvolle, 
# kun taas toisin päin ei tule ongelmaa sillä aina on menneisyydessä arvo...
"""
def correct_volume_based_on_lowest_elevation():
    # aloita tämän hetken volyymistä ja tee siitä eteenpäin, sillä silloin on jo suora korkeus.
    # Ekalle tunnille erotus kertaa tuuli suunnan asetus ja sitten lopuille tehdään niin, että laitetaan oikeisiin suuntiin asiat yksinkertaisemmilla kertoimilla.
    # Laske alimmat kohdat jokaiselle alueen sijainnille niille kohdille, jotka tutka havaitsee.
    
    # tehdään vaakasuunnassa lineaarinen interpolointi

"""