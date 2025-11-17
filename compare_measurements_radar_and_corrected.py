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
    "Ranua_Aho.csv": (66.1502112, 26.10784477),
    "Ranua_Lentokenttä": (65.977172, 26.367571),
    "Tornio_Torppi.csv": (65.84733, 24.17369),
    #"Hailuoto_Keskikylä.csv": (65.02, 24.73),
    #"Kajaani_Petäisenniska.csv": (64.22, 27.75),
    #"Oulu_Kaukovainio.csv": (65, 25.52),
    #"Siikajoki_Ruukki.csv": (64.68, 25.09),
    #"Oulu_Oulunsalo_Pellonpää.csv", lakkautettu
    #"Pudasjärvi_lentokenttä.csv": (65.4, 26.96),
    #"Vaala_Pelso.csv": (64.5, 26.42),
    #"Puolanka_Paljakka.csv": (64.66, 28.06),
    #'C12503.csv': ( 65.004402 , 25.507365 ),
    #'C12504.csv': ( 65.026835 , 25.507024 ),
    #'C12513.csv': ( 65.012021 , 25.506451 ),
    #'C12552.csv': ( 64.985992 , 25.50536 ),
    #'C12559.csv': ( 65.029746 , 25.527267 ),
    #'C12560.csv': ( 65.037447 , 25.551205 ),
    #'C12570.csv': ( 64.955832 , 25.504861 ),
    #'C12581.csv': ( 65.040836 , 25.474185 ),
    #'C12587.csv': ( 65.078872 , 25.441991 ),
    #'C12592.csv': ( 65.005286 , 25.504239 ),
    #'C12593.csv': ( 65.003887 , 25.510597 ),
    #'C12594.csv': ( 64.991107 , 25.537897 ),
    #'C12595.csv': ( 64.980371 , 25.561628 ),
    #'C12597.csv': ( 65.063158 , 25.51173 ),
    #'C12598.csv': ( 65.005824 , 25.579649 ),
    #'C12612.csv': ( 65.019907 , 25.507899 ),
    #'C12613.csv': ( 64.940686 , 25.5387 ),
    #'C12614.csv': ( 64.959079 , 25.517196 ),
    #'C12615.csv': ( 65.055091 , 25.44914 ),
    #'C12617.csv': ( 64.920607 , 25.534802 ),
}

outputs = ["corrected","uncorrected"]

datatypes = ['DBZH',"HCLASS"]

shape = None

first_date = datetime(2024,11,20,12,0)
for datatype in datatypes:
    if datatype == 'HCLASS':
        filenames = ["./corrected_field_HCLASS/{year:02d}/{month:02d}/{day:02d}/{year:02d}{month:02d}{day:02d}{hour:02d}{minute:02d}.h5",
                    "/arch/radar/HDF5/{year:02d}/{month:02d}/{day:02d}/radar/polar/fiuta/{year:02d}{month:02d}{day:02d}{hour:02d}{minute:02d}_radar.polar.fiuta.h5"]
    else:
        filenames = ["./corrected_field/{year:02d}/{month:02d}/{day:02d}/{year:02d}{month:02d}{day:02d}{hour:02d}{minute:02d}.h5",
                    "/arch/radar/HDF5/{year:02d}/{month:02d}/{day:02d}/radar/polar/fiuta/{year:02d}{month:02d}{day:02d}{hour:02d}{minute:02d}_radar.polar.fiuta.h5"]
        
    end_date = datetime(2025,2,14,16,0)
    for clas, outpute in enumerate(outputs):
        radar_filename = filenames[clas]
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
                    try:
                        pvol = xr.open_dataset(radar_h5_file, group="sweep_0", engine="odim")
                        moment = pvol.wrl.georef.georeference(
                                    crs=wrl.georef.get_default_projection()
                                )
                        value = moment[datatype][az_ran].data 
                        # Lisää vain jos on järkevä luku
                        if value < 150:
                            radar_measurements.append((date, value))
                    except:
                        print("An exception occurred")
                        with open("error_while_reading_h5.txt", 'a') as file:
                            file.write(f"{date.strftime('%Y-%m-%d-%H-%M')}\n")
                date += timedelta(minutes=5)
                #print(pvol["sweep_0"].ds.wrl.georef.georeference(
                #    crs=wrl.georef.get_default_projection()
                #)['DBZH'].data)
                #selected_data = pvol.sel(x=lat, y=lon, method='nearest')
                #print(selected_data)
                #exit()
            #tallenna tuo lista filuksi!
            df = pd.DataFrame(radar_measurements, columns=['Date', 'Value'])
            if datatype == 'HCLASS':
                filename_out = outpute+'_HCLASS_at_aws/'+ filename
            else:
                filename_out = outpute+'_at_aws/'+ filename
            
            # Write to CSV
            df.to_csv(filename_out, index=False)


            # Kerää annetun latlonin perusteella aineisto tutkavolyymeistä ja myös korjatuista
            # Avaa tutkadata
            # Katso sijainnista mikä on arvo latlonissa
            # Kerää datapiste
            # Tallenna datapiste latlonille tarkoitettuun listaan
        # Lopuksi tallenna listat
