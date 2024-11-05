import os
import requests
import xarray as xr

def download_file_as_temp(url, directory='smartmets'):
    temp_path = os.path.join(directory, 'latest_temp_smartmet.grib')

    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print("File downloaded and saved as 'latest_temp_smartmet.grib'")
            latest_path = os.path.join(directory, 'latest_smartmet.grib')
            if os.path.exists(latest_path):
                latest_ds = xr.open_dataset(latest_path, engine='cfgrib')
                temp_ds = xr.open_dataset(temp_path, engine='cfgrib')
                if temp_ds.time.data <= latest_ds.time.data:
                    print(temp_ds.time.data, latest_ds.time.data)
                    temp_ds.close()
                    os.remove(temp_path)
                    print("removing temp_ds due to same or earlier generation timestamp as the timestamp for latest_ds")

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print("An error occurred during download or saving:", str(e))

def rename_files(directory='smartmets'):
    latest_path = os.path.join(directory, 'latest_smartmet.grib')
    second_path = os.path.join(directory, 'second_latest_smartmet.grib')
    oldest_path = os.path.join(directory, 'oldest_smartmet.grib')
    temp_path = os.path.join(directory, 'latest_temp_smartmet.grib')

    if os.path.exists(temp_path):
        # tässä päivitetään, jos lähtö osoite on olemassa
        if os.path.exists(second_path):
            os.rename(second_path, oldest_path)
        else:
            print(second_path, "did not exist")
    
        if os.path.exists(latest_path):
            os.rename(latest_path, second_path)
        else:
            print(latest_path, "did not exist")
    
        os.rename(temp_path, latest_path)
    else:
        print(temp_path, "does not exist")

if __name__ == '__main__':

    url = "http://smartmet.fmi.fi/download?producer=harmonie_skandinavia_mallipinta&bbox=15,58,45,70&param=temperature,WindVMS,WindUMS&timestep=60&levels=18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65&timesteps=9&format=grib2"
    
    download_file_as_temp(url)
    rename_files()