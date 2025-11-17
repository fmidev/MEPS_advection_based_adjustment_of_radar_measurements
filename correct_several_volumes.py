from correct_radar_volume import *
from datetime import datetime, timedelta


starttime = datetime(2024,11,20,12,0)
endtime = datetime(2025,3,18,6,0)
curtime = starttime 

while curtime <= endtime:
    cur_str = curtime.strftime("%Y%m%d%H%M")
    print(cur_str)
    
    correct_radar_volumes(cur_str, field='HCLASS')
    curtime += timedelta(hours=3)


