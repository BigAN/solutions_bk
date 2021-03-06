nohup python -u 9778_data_produce.py --debug f --subsample f  --nchunk 179903890 --val 4000000 >> basicfea
python -u 9778_data_produce__prev_click.py --debug f --subsample f   --nchunk 179903890 --val 4000000  --fname prev >> prev
python -u 9778_data_produce__diff_feature.py --debug f --subsample f  --nchunk 179903890 --val 4000000 --fname diff >> diff
python -u 9778_data_produce__more_unique.py --debug f --subsample f  --nchunk 179903890 --val 4000000 --fname unique >> unique
python -u 9778_data_produce__more2_unique.py --debug f --subsample f  --nchunk 179903890 --val 4000000 --fname unique2 >> unique2


nohup python -u 9778_train_ftr_avg.py --fileno avg_1 --tra train_1_179903891_False_False.ftr --tst test_1_179903891_False_False.ftr --val val_1_179903891_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--rm ip_app_os_by_hour_var,ip_app_oscount,ip_by_os_cumcount,ip_channel_prevClick,ip_os_prevClick,ip_day_by_hour_countuniq \
--addfeas prev,diff,unique,unique2 \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff#next_next_ip_os_device_app_diff_mean#next_next_ip_app_diff#next_prev_ip_app_diff,\
ip_by_os_countuniq#ip_app_channel_by_device_countuniq#ip_device_os_appcount#app_channelcount#ip_oscount#ipcount#ip_app_by_device_cumcount#app_channel_by_ip_countuniq#app_channel_by_os_countuniq,\
channel_by_app_countuniq#app_oscount#app_os_by_channel_countuniq#app_os_by_device_countuniq#app_os_by_ip_countuniq\
>> avg_9778_rs 2>&1 &

nohup python -u 9778_train_ftr_avg.py --fileno avg_full --tra train_1_179903891_False_False.ftr --tst test_1_179903891_False_False.ftr --val val_1_179903891_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--rm ip_app_os_by_hour_var,ip_app_oscount,ip_by_os_cumcount,ip_channel_prevClick,ip_os_prevClick,ip_day_by_hour_countuniq \
--addfeas prev,diff,unique,unique2 \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff#next_next_ip_os_device_app_diff_mean#next_next_ip_app_diff#next_prev_ip_app_diff#next_3_ip_os_device_app_diff#next_6mins_ip_os_device_app_diff#next_3_ip_app_diff#next_6mins_ip_app_diff,\
ip_by_os_countuniq#ip_app_channel_by_device_countuniq#ip_device_os_appcount#app_channelcount#ip_oscount#ipcount#ip_app_by_device_cumcount#app_channel_by_ip_countuniq#app_channel_by_os_countuniq,\
channel_by_app_countuniq#app_oscount#app_os_by_channel_countuniq#app_os_by_device_countuniq#app_os_by_ip_countuniq\
>> avg_9778_rs 2>&1 &

