
# \
python -u 9778_train_ftr.py --tra train_119903890_179903890_False_False.ftr --tst test_119903890_179903890_False_False.ftr --val val_119903890_179903890_False_False.ftr \
--fea app,channel,device,os,hour,ip_app_device_os_channel_prevClick,ip_os_device_prevClick,ip_os_device_app_prevClick,ip_app_device_os_channel_nextClick,ip_os_device_nextClick,ip_os_device_app_nextClick,ip_by_channel_countuniq,ip_device_os_by_app_countuniq,ip_day_by_hour_countuniq,ip_by_app_countuniq,ip_app_by_os_countuniq,ip_by_device_countuniq,app_by_channel_countuniq,ip_by_os_cumcount,ip_device_os_by_app_cumcount,ip_day_hourcount,ip_appcount,ip_app_oscount,ip_day_channel_by_hour_var,ip_app_os_by_hour_var,ip_app_channel_by_day_var,ip_app_channel_by_hour_mean \
--rm ip_app_device_os_channel_prevClick,ip_os_device_app_prevClick,ip_os_device_prevClick,ip_day_channel_by_hour_var \
--addfeas prev,lagfea \
--detailfeas ip_channel_prevClick#ip_os_prevClick,first_ip_app_lagFea#last_ip_app_lagFea#first_ip_os_lagFea#last_ip_os_lagFea#first_ip_app_device_os_channel_lagFea#last_ip_app_device_os_channel_lagFea#first_ip_os_device_lagFea#last_ip_os_device_lagFea#first_ip_os_device_app_lagFea#last_ip_os_device_app_lagFea

# \