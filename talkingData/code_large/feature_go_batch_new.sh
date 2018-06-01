python -u 9778_train_ftr.py --fileno prev --tra train_119903890_144903890_False_False.ftr --tst test_119903890_144903890_False_False.ftr --val val_119903890_144903890_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--addfeas prev \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick \
>> 9778_rs

python -u 9778_train_ftr.py --fileno prev_lagfea --tra train_119903890_144903890_False_False.ftr --tst test_119903890_144903890_False_False.ftr --val val_119903890_144903890_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--addfeas prev,lagfea \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
last_ip_app_lagFea#last_ip_os_device_app_lagFea,\
>> 9778_rs

python -u 9778_train_ftr.py --fileno prev_lagfea_diff --tra train_119903890_144903890_False_False.ftr --tst test_119903890_144903890_False_False.ftr --val val_119903890_144903890_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--addfeas prev,lagfea,diff \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
last_ip_app_lagFea#last_ip_os_device_app_lagFea,\
prev_first_ip_app_diff#prev_last_ip_app_diff#prev_max_ip_app_diff#prev_min_ip_app_diff#prev_std_ip_app_diff#prev_prev_ip_app_diff#prev_next_ip_app_diff#next_first_ip_app_diff#next_last_ip_app_diff#next_max_ip_app_diff#next_min_ip_app_diff#next_std_ip_app_diff#next_prev_ip_app_diff#next_next_ip_app_diff#prev_first_ip_os_device_app_diff#prev_last_ip_os_device_app_diff#prev_max_ip_os_device_app_diff#prev_min_ip_os_device_app_diff#prev_std_ip_os_device_app_diff#prev_prev_ip_os_device_app_diff#prev_next_ip_os_device_app_diff#next_first_ip_os_device_app_diff#next_last_ip_os_device_app_diff#next_max_ip_os_device_app_diff#next_min_ip_os_device_app_diff#next_std_ip_os_device_app_diff#next_prev_ip_os_device_app_diff#next_next_ip_os_device_app_diff\
>> 9778_rs