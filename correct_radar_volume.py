
        # Tästä alkaa keräysfunktio!
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
   
