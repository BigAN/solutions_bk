#python -u 9778_train_ftr.py --tra train_149903890_184903890_False_False.ftr --tst test_149903890_184903890_False_False.ftr --val val_149903890_184903890_False_False.ftr \
#--fea app,channel,device,os,hour,minute,second,ip_by_channel_countuniq,ip_device_os_by_app_countuniq,ip_day_by_hour_countuniq,ip_by_app_countuniq,ip_app_by_os_countuniq,ip_by_device_countuniq,app_by_channel_countuniq,ip_by_os_cumcount,ip_device_os_by_app_cumcount,ip_day_hourcount,ip_appcount,ip_app_oscount,ip_day_channel_by_hour_var,ip_app_os_by_hour_var,ip_app_channel_by_day_var,ip_app_channel_by_hour_mean \
#--rm ip_day_channel_by_hour_var,ip_app_channel_by_day_var,ip_app_channel_by_hour_mean,minute,second,ip_app_os_by_hour_var \
#--addfeas lagfea,prev,diff \
#--detailfeas last_ip_app_lagFea#last_ip_os_device_app_lagFea#last_ip_app_device_os_channel_lagFea#first_ip_os_device_lagFea#last_ip_os_device_lagFea,\
#ip_os_device_app_nextClick#ip_os_device_nextClick#ip_app_device_os_channel_nextClick,\
#next_next_ip_os_device_app_diff#next_next_ip_app_diff#prev_next_ip_os_device_app_diff


# base local 0.977364 lb
nohup python -u 9778_train_ftr.py --fileno prev --tra train_149903890_184903890_False_False.ftr --tst test_149903890_184903890_False_False.ftr --val val_149903890_184903890_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--addfeas prev \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick \
>> 9778_rs 2>&1 &
#--rm ip_day_channel_by_hour_var,ip_app_channel_by_day_var,ip_app_channel_by_hour_mean,minute,second,ip_app_os_by_hour_var \


# add lag fea,0.977803
nohup python -u 9778_train_ftr.py --fileno prev_lagfea --tra train_149903890_184903890_False_False.ftr --tst test_149903890_184903890_False_False.ftr --val val_149903890_184903890_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--addfeas prev,lagfea \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
last_ip_app_lagFea#last_ip_os_device_app_lagFea,\
>> 9778_rs 2>&1 &

#add diff ,
nohup python -u 9778_train_ftr.py --fileno prev_lagfea_diff --tra train_149903890_184903890_False_False.ftr --tst test_149903890_184903890_False_False.ftr --val val_149903890_184903890_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--addfeas prev,lagfea,diff \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
last_ip_app_lagFea#last_ip_os_device_app_lagFea,\
next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff\
>> 9778_rs 2>&1 &

nohup python -u 9778_train_ftr.py --fileno prev_unique_diff --tra train_149903890_184903890_False_False.ftr --tst test_149903890_184903890_False_False.ftr --val val_149903890_184903890_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--addfeas prev,unique,diff \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
ip_by_os_countuniq#ip_day_by_period_countuniq#ip_device_by_channel_countuniq#ip_app_by_device_countuniq#ip_device_os_app_by_period_countuniq,\
next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff\
>> 9778_rs 2>&1 &
#ip_os_by_device_countuniq rm
#ip_device_by_channel_countuniq good


#test
nohup python -u 9778_train_ftr.py --fileno moreuniq --tra train_149903890_184903890_False_False.ftr --tst test_149903890_184903890_False_False.ftr --val val_149903890_184903890_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--rm ip_app_os_by_hour_var,ip_app_oscount,ip_by_os_cumcount,ip_channel_prevClick,ip_os_prevClick,ip_day_by_hour_countuniq \
--addfeas prev,unique,diff \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
ip_by_os_countuniq#ip_day_by_period_countuniq#ip_device_by_channel_countuniq#ip_device_by_period_countuniq#ip_device_by_channel_cumcount#ip_app_by_device_countuniq#ip_os_by_device_countuniq#ip_device_os_app_by_period_countuniq,\
next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff\
>> 9778_rs 2>&1 &
