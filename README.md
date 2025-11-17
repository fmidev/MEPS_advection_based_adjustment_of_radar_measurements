# MEPS_advection_based_adjustment_of_radar_measurements
Implementation of Advection-Based Adjustment of Radar Measurements for MEPS
Original paper DOI: https://doi.org/10.1175/MWR-D-11-00045.1 

# Program
The program is run by the shell script collect_and_combine_smartmet.sh

After setting up the right environment and directory, it first sets up the current time to be 3 hours ago
Based on this time scripts for colecting computing the advection correction mapping and the correction itself.
These are done in scripts compute_advection_correction.py and correct_radar_volume.py.
Then the smartmet NWP from current timestamp is collected and the old ones are removed in script download_current_and_remove_oldest.py
Finally the script finishes with combine_the_latest_smartmet_files.py to create the new file for making predictions. In order to have the best available NWP data.

# The mapping
The mapping class in script compute_advection_correction.py
The class method get_adjusted mapping does the magic. It works by opening the a reference file first in order to compute the mapping for a corresponding radar.

As the computation is run for the last three hours in order to have the decent data available. The 3 hours old MEPS is used for the timestamps in the following three hours. If there is need for data further than 3 hours into the past the 6 hours old MEPS will be used.

These timestamps will be iterated and a mapping file will be computed for each timestamp. The latlon pairs from the reference file and the ground elevation file will be used to map the ground location of a radar bin. From this ground location the correction algorithm will move to past in time and higher in altitude and to different location in lat lon coordinates. These mappings from the lat lons to the the altitude of the lowest radar bin will be computed in parallel using the advection_from_a_grid_cell method.

After having obtained the data for each timestep latitude longitude and bin height. These values computed will be interpolated. This is done because the advection from a grid cell will be computed for only so many location for efficiency. The interpolation will be done using azimuth range coordinates and will produce the maps for each azimuth and range pair to the corresponding latitude and longitude in the past from which location the new radar value should be extracted.

## advection_from_a_grid_cell
The advection_from_a_grid_cell method computes the trajectories using data from the numerical weather prediction. It uses the u and v winds to compute the horizontal displacement and the update rule to compute the vertical change in time. The vertical change rule needs information from the height of the melting layer which is the height of the 0 temperature values in the model above the current location. 

Starting from the ground location and given the starting time, adaptive timestep is calculated by selecting the value corresponding to the smallest change in the grid vertically or horizontally. Based on these rules the location will be eventually mapped to the lowest elevation angle radar bin, by moving the latlon values, the elevation, and the time.