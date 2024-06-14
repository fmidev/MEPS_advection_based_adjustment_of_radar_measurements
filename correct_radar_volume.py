# alimmat kulmat PPI1_A, PPI2_A ja PPI3_A:n  matala kulma (0.3) näkee kauas (Luostolla 0.1).
# 250 km maanpinnalla ollaan korkeudessa noin 5 km

# alin smartmet on 12 m
# korkeus_malli_mappi

# Olisi varmaan hyvä olla height to level sittenkin.
import xarray as xr
import rioxarray
import numpy as np
import wradlib as wrl


class advection_adjustment():
    def __init__(self):
        self.ground_level_dataset = xr.open_dataset("suomi_korkeusalueet_luvuilla.tif", decode_coords="all").to_dataarray()
        self.weather = xr.open_dataset("latest_smartmet.grib", engine="cfgrib", decode_coords="all").to_dataarray()
        self.height_to_level = {5320.3: 24, 4997.4: 25, 4688.0: 26, 4391.7: 27, 4108.1: 28, 3836.9: 29, 3577.8: 30, 3330.4: 31, 3094.5: 32, 2870.0: 33, 2656.5: 34, 2454.1: 35, 2262.5: 36, 2081.5: 37, 1911.2: 38, 1751.4: 39, 1602.1: 40, 1463.1: 41, 1334.4: 42, 1215.0: 43, 1104.4: 44, 1001.8: 45, 906.8: 46, 819.0: 47, 737.8: 48, 662.9: 49, 593.8: 50, 530.1: 51, 471.5: 52, 417.6: 53, 368.1: 54, 322.6: 55, 280.7: 56, 242.2: 57, 206.6: 58, 173.7: 59, 143.1: 60, 114.5: 61, 87.5: 62, 61.7: 63, 36.8: 64, 12.2: 65}
        level_to_height = {24: 5320.3, 25: 4997.4, 26: 4688.0, 27: 4391.7, 28: 4108.1, 29: 3836.9, 30: 3577.8, 31: 3330.4, 32: 3094.5, 33: 2870.0, 34: 2656.5, 35: 2454.1, 36: 2262.5, 37: 2081.5, 38: 1911.2, 39: 1751.4, 40: 1602.1, 41: 1463.1, 42: 1334.4, 43: 1215.0, 44: 1104.4, 45: 1001.8, 46: 906.8, 47: 819.0, 48: 737.8, 49: 662.9, 50: 593.8, 51: 530.1, 52: 471.5, 53: 417.6, 54: 368.1, 55: 322.6, 56: 280.7, 57: 242.2, 58: 206.6, 59: 173.7, 60: 143.1, 61: 114.5, 62: 87.5, 63: 61.7, 64: 36.8, 65: 12.2}
        self.shape = self.weather['latitude'].values.shape
        melting_layer_height= (np.abs(self.weather.sel(variable='t')-273.15).argmin(dim='hybrid'))+24
        melting_layer_height.data = np.vectorize(level_to_height.get)(melting_layer_height)
        self.melting_layer_height=melting_layer_height
        self.heights  = np.array(list(self.height_to_level.keys()))
        self.flat_lons = self.weather['longitude'].values.flatten()
        self.flat_lats = self.weather['latitude'].values.flatten()
    
    def get_closest_xy_coordinate_in_model(self,x,y):
        closest = np.sqrt((self.flat_lats - y)**2 + (self.flat_lons - x)**2).argmin()
        return np.unravel_index(closest, self.shape)

    def get_melting_layer_height(self,x,y,step=0):
        closest_y, closest_x = self.get_closest_xy_coordinate_in_model(x,y)
        return self.melting_layer_height.isel(y=closest_y,x=closest_x, step=step).data

    # nearest value selection
    def get_model_level(self,z):
        index = np.abs(self.heights - z)
        smallest_value = self.heights[np.argmin(index)]
        return self.height_to_level[smallest_value]

    def get_radar_bin_heigth(x,y, radar_antenna):
        return

    def get_fall_speed(self,z,melting_layer):
        return 1 + min(4, (4/700)*max(z-(melting_layer-700),0)) #m/s

    def get_weather_data(self,x,y,z,step=0):
        closest_y, closest_x = self.get_closest_xy_coordinate_in_model(x,y)
        return self.weather.isel(y=closest_y,x=closest_x, step=step).sel(hybrid=self.get_model_level(z))

    def advection_from_a_grid_cell(self,x,y):
        z = self.ground_level_dataset.sel(y=y,x=x,method="nearest").data[0,0]
        el_h = self.get_radar_bin_height(x,y)
        #Yksi iteraation on sekunti
        while el_h > z:
            temp_and_wind = self.get_weather_data(x,y,z,step=0)
            x -= temp_and_wind.sel(variable='u').data
            y -= temp_and_wind.sel(variable='v').data
            ml_height = self.get_melting_layer_height(x,y,step=0)
            z += self.get_fall_speed(z,ml_height)
            el_h = self.get_radar_bin_height(x,y)
        return (x,y,z)

    def get_adjusted_dbz(self):
        # avataan alkuun geneerinen radar compo ja siitä lasketaan binien sijainnit maan pinnalla.
        # Sitten valitaan jokainen x ja y sijainti tästä listasta ja ruvetaan laskemaan...
        # arvioidaa


import xradar as xd

filename = "202208281555_fianj_PVOL.h5"

pvol = xd.io.open_odim_datatree(filename)
pvol['sweep_0']

# nykyhetkestä ja mennä menneeseen koska menneisyydestä tulevaisuuteen voidaan joutua mappaamaan samalle arvolle, 
# kun taas toisin päin ei tule ongelmaa sillä aina on menneisyydessä arvo...
"""
def correct_volume_based_on_lowest_elevation():
    # aloita tämän hetken volyymistä ja tee siitä eteenpäin, sillä silloin on jo suora korkeus.
    # Ekalle tunnille erotus kertaa tuuli suunnan asetus ja sitten lopuille tehdään niin, että laitetaan oikeisiin suuntiin asiat yksinkertaisemmilla kertoimilla.
    # Laske alimmat kohdat jokaiselle alueen sijainnille niille kohdille, jotka tutka havaitsee.
    
    # tehdään vaakasuunnassa lineaarinen interpolointi

"""