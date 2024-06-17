import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling


src = rio.open("suomi.tif")
array = src.read()
metadata = src.meta
# tässä päivitetään tietotyyppi ja bandmäärä
metadata.update(count=1, dtype='float32')
map_colors_to_values_vrt_sea = {(117, 173, 87): 12.25,#0.5-25 
 (235, 252, 119): 125,# 100-150
 (200, 247, 119): 87.5,# 75-100
 (252, 243, 109):175,# 150-200
 (204, 152, 61):375,# 350-400
 (171, 245, 122):62.5, #50-75
 (190, 210, 255):-5.5,#-10-0.5 
 (89, 71, 51): 1200,# 1000-1400
 (145, 214, 107):42.5, # 25-50
 (255, 255, 255): None,# None
 (252, 221, 93):225,# 200-250
 (176, 135, 55):500,# 400-600
 (148, 115, 58):700,# 600-800
 (50, 100, 255):-105,# -200--10
 (117, 92, 56):900,# 800-1000
 (232, 168, 65):325,# 300-350
 (250, 189, 75):275}# 250-300

map_colors_to_index = {(117, 173, 87): 0,#0.5-25 
 (235, 252, 119): 1,# 100-150
 (200, 247, 119): 2,# 75-100
 (252, 243, 109):3,# 150-200
 (204, 152, 61):4,# 350-400
 (171, 245, 122):5, #50-75
 (190, 210, 255):6,#-10-0.5 
 (89, 71, 51):7,# 1000-1400
 (145, 214, 107):8, # 25-50
 (255, 255, 255):9,# None
 (252, 221, 93):10,# 200-250
 (176, 135, 55):11,# 400-600
 (148, 115, 58):12,# 600-800
 (50, 100, 255):13,# -200--10
 (117, 92, 56):14,# 800-1000
 (232, 168, 65):15,# 300-350
 (250, 189, 75):16}# 250-300

dst_crs = 'EPSG:4326'

transform, width, height = calculate_default_transform(
    src.crs, dst_crs, src.width, src.height, *src.bounds)
#py, px = dataset.index(lon, lat)

new_array = -np.ones((array.shape[1],array.shape[2]))

for i in range(array.shape[1]):
    for j in range(array.shape[2]):

        osanen = array[:,i,j]

        new_array[i,j] = map_colors_to_values_vrt_sea[(osanen[0],osanen[1],osanen[2])]
metadata.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
print(metadata)

with rio.open("suomi_korkeusalueet_luvuilla.tif", 'w', **metadata) as dst:
    reproject(
        source=new_array,
        destination=rio.band(dst, 1),
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest)

src.close()