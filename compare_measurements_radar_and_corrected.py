import pandas as pd
from datetime import datetime,timedelta
import xradar as xd
import xarray as xr
import wradlib as wrl
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# Nämä on Lat Lon
file_list = {
    "Hailuoto_Keskikylä.csv": (65.02, 24.73),
    "Kajaani_Petäisenniska.csv": (64.22, 27.75),
    "Oulu_Kaukovainio.csv": (65, 25.52),
    "Siikajoki_Ruukki.csv": (64.68, 25.09),
    #"Oulu_Oulunsalo_Pellonpää.csv", lakkautettu
    "Pudasjärvi_lentokenttä.csv": (65.4, 26.96),
    "Vaala_Pelso.csv": (64.5, 26.42),
    "Puolanka_Paljakka.csv": (64.66, 28.06)
}

radar_filename = "/home/myllykos/Documents/programming/MEPS_advection_based_adjustment_of_radar_measurements/corrected_field/{year:02d}/{month:02d}/{day:02d}/{year:02d}{month:02d}{day:02d}{hour:02d}{second:02d}.h5"

shape = None

first_date = datetime(2024,11,19,6,0)

end_date = datetime(2025,2,14,16,0)
end_date= datetime(2025,11,19,7,0)
radar_h5_file =radar_filename.format(
                        year=first_date.year,
                        month=first_date.month,
                        day=first_date.day,
                        hour=first_date.hour,
                        minute=first_date.minute,
                        second=first_date.second,
                    )
pvol = xr.open_dataset(radar_h5_file, group="sweep_0", engine="odim")
#pvol = xd.io.open_odim_datatree(radar_h5_file)
moment = pvol.wrl.georef.georeference(
            crs=wrl.georef.get_default_projection()
        )
cKDTree_w = cKDTree(np.column_stack((moment.y.data.flatten(), moment.x.data.flatten())))
shape = moment.y.data.shape

def get_nearest_xy_coordinate_in_model(x,y):
    dist, idx = cKDTree_w.query((y, x))
    return np.unravel_index(idx, shape)

cKDTree_weather = None
for filename in file_list.keys():
    #a = pd.read_csv("/home/myllykos/Desktop/utajärvi_havainnot/"+filename)
    lat,lon = file_list[filename]
    radar_measurements = []
    date = first_date
    
    az_ran=get_nearest_xy_coordinate_in_model(lon,lat)
    while date <= end_date:
        print(date, flush=True)
        radar_h5_file =Path(radar_filename.format(
                        year=date.year,
                        month=date.month,
                        day=date.day,
                        hour=date.hour,
                        minute=date.minute,
                        second=date.second,
                    ))
        
        if radar_h5_file.is_file():
            pvol = xr.open_dataset(radar_h5_file, group="sweep_0", engine="odim")
            moment = pvol.wrl.georef.georeference(
                        crs=wrl.georef.get_default_projection()
                    )
            value = moment['DBZH'][az_ran].data 
            # Lisää vain jos on järkevä luku
            if value < 150:
                radar_measurements.append((date, value))
        date += timedelta(minutes=5)
        #print(pvol["sweep_0"].ds.wrl.georef.georeference(
        #    crs=wrl.georef.get_default_projection()
        #)['DBZH'].data)
        #selected_data = pvol.sel(x=lat, y=lon, method='nearest')
        #print(selected_data)
        #exit()
    #tallenna tuo lista filuksi!
    df = pd.DataFrame(radar_measurements, columns=['Date', 'Value'])
    filename_out = 'advection_corrected_at_aws/'+ filename

    # Write to CSV
    df.to_csv(filename_out, index=False)


    # Kerää annetun latlonin perusteella aineisto tutkavolyymeistä ja myös korjatuista
    # Avaa tutkadata
    # Katso sijainnista mikä on arvo latlonissa
    # Kerää datapiste
    # Tallenna datapiste latlonille tarkoitettuun listaan
# Lopuksi tallenna listat