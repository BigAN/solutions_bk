#python -u 9778_train_ftr.py --tra train_119903890_179903890_False_False.ftr --tst test_119903890_179903890_False_False.ftr --val val_119903890_179903890_False_False.ftr \
#--fea app,channel,device,os,hour,minute,second,ip_by_channel_countuniq,ip_device_os_by_app_countuniq,ip_day_by_hour_countuniq,ip_by_app_countuniq,ip_app_by_os_countuniq,ip_by_device_countuniq,app_by_channel_countuniq,ip_by_os_cumcount,ip_device_os_by_app_cumcount,ip_day_hourcount,ip_appcount,ip_app_oscount,ip_day_channel_by_hour_var,ip_app_os_by_hour_var,ip_app_channel_by_day_var,ip_app_channel_by_hour_mean \
#--rm ip_day_channel_by_hour_var,ip_app_channel_by_day_var,ip_app_channel_by_hour_mean,minute,second,ip_app_os_by_hour_var \
#--addfeas lagfea,prev,diff \
#--detailfeas last_ip_app_lagFea#last_ip_os_device_app_lagFea#last_ip_app_device_os_channel_lagFea#first_ip_os_device_lagFea#last_ip_os_device_lagFea,\
#ip_os_device_app_nextClick#ip_os_device_nextClick#ip_app_device_os_channel_nextClick,\
#next_next_ip_os_device_app_diff#next_next_ip_app_diff#prev_next_ip_os_device_app_diff


# base local 0.977364 lb
#nohup python -u 9778_train_ftr.py --fileno prev --tra train_119903890_179903890_False_False.ftr --tst test_119903890_179903890_False_False.ftr --val val_119903890_179903890_False_False.ftr \
#--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
#--addfeas prev \
#--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick \
#>> 9778_rs 2>&1 &
#--rm ip_day_channel_by_hour_var,ip_app_channel_by_day_var,ip_app_channel_by_hour_mean,minute,second,ip_app_os_by_hour_var \


# add lag fea,0.977803
#nohup python -u 9778_train_ftr.py --fileno prev_lagfea --tra train_119903890_179903890_False_False.ftr --tst test_119903890_179903890_False_False.ftr --val val_119903890_179903890_False_False.ftr \
#--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
#--addfeas prev,lagfea \
#--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
#last_ip_app_lagFea#last_ip_os_device_app_lagFea,\
#>> 9778_rs 2>&1 &

#add diff ,
#nohup python -u 9778_train_ftr.py --fileno prev_lagfea_diff --tra train_119903890_179903890_False_False.ftr --tst test_119903890_179903890_False_False.ftr --val val_119903890_179903890_False_False.ftr \
#--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
#--addfeas prev,lagfea,diff \
#--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
#last_ip_app_lagFea#last_ip_os_device_app_lagFea,\
#next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff\
#>> 9778_rs 2>&1 &

#nohup python -u 9778_train_ftr_val.py --fileno prev_unique --tra train_119903890_179903890_False_False.ftr --tst test_119903890_179903890_False_False.ftr --val val_119903890_179903890_False_False.ftr \
#--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_by_os_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
#--addfeas prev,unique \
#--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
#ip_by_os_countuniq#ip_day_by_period_countuniq#ip_device_by_channel_countuniq#ip_app_by_device_countuniq#ip_device_os_app_by_period_countuniq \
#>> 9778_rs 2>&1 &
#ip_os_by_device_countuniq rm
#ip_device_by_channel_countuniq good


#test add app_os_by_channel_countuniq,channel_by_app_countuniq
# add next_next_ip_app_diff_mean#next_next_ip_app_diff
#nohup python -u 9778_train_ftr.py --fileno moreuniq3 --tra train_119903890_179903890_False_False.ftr --tst test_119903890_179903890_False_False.ftr --val val_119903890_179903890_False_False.ftr \
#--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
#--rm ip_app_os_by_hour_var,ip_app_oscount,ip_by_os_cumcount,ip_channel_prevClick,ip_os_prevClick,ip_day_by_hour_countuniq \
#--addfeas prev,diff,unique \
#--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
#next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff#next_next_ip_app_diff#next_prev_ip_app_diff#next_next_ip_os_diff_mean,\
#ip_by_os_countuniq#ip_app_channel_by_device_countuniq#ip_device_os_appcount#app_channelcount#ip_oscount#ipcount#ip_app_by_device_cumcount#app_channel_by_ip_countuniq#app_channel_by_os_countuniq,\
#>> 9778_rs 2>&1 &
#channel_by_app_countuniq#app_oscount#app_os_by_channel_countuniq#app_os_by_device_countuniq#app_os_by_ip_countuniq \

#0.98408
# high 相关度 next_next_ip_app_diff_mean ,带来不了提升.
python -u 9778_data_produce__diff_feature.py --debug f --subsample f  --nchunk 60000000 --val 4000000 --fname diff >> diff

nohup python -u find_paras_9778_train_ftr.py --fileno moreuniq6 --tra train_119903890_179903890_False_False.ftr --tst test_119903890_179903890_False_False.ftr --val val_119903890_179903890_False_False.ftr \
--fea ip_by_app_countuniq,hour,channel,ip_device_os_by_app_cumcount,app,ip_by_device_countuniq,app_by_channel_countuniq,ip_day_hourcount,ip_by_channel_countuniq,ip_app_os_by_hour_var,device,ip_device_os_by_app_countuniq,ip_appcount,os,ip_by_os_cumcount,ip_day_by_hour_countuniq,ip_app_oscount \
--rm ip_app_os_by_hour_var,ip_app_oscount,ip_by_os_cumcount,ip_channel_prevClick,ip_os_prevClick,ip_day_by_hour_countuniq \
--addfeas prev,diff,unique,unique2 \
--detailfeas ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,\
next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff#next_next_ip_os_device_app_diff_mean#next_next_ip_app_diff#next_prev_ip_app_diff,\
ip_by_os_countuniq#ip_app_channel_by_device_countuniq#ip_device_os_appcount#app_channelcount#ip_oscount#ipcount#ip_app_by_device_cumcount#app_channel_by_ip_countuniq#app_channel_by_os_countuniq,\
channel_by_app_countuniq#app_oscount#app_os_by_channel_countuniq#app_os_by_device_countuniq#app_os_by_ip_countuniq\
>> hpyer_rs 2>&1 &

#origin
#next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff#next_next_ip_os_device_app_diff_mean#next_next_ip_app_diff#next_prev_ip_app_diff,\
#new diff 0.9837
#next_next_ip_app_diff_mean#next_next_ip_os_device_app_diff#next_next_ip_app_diff#next_prev_ip_os_device_app_diff#next_next_ip_os_app_diff#next_prev_ip_os_app_diff



#origin
#ip_channel_prevClick#ip_os_prevClick#ip_app_device_os_channel_nextClick#ip_os_device_app_nextClick#ip_os_device_nextClick,
#ip_by_os_countuniq#ip_app_channel_by_device_countuniq#ip_device_os_appcount#app_channelcount#ip_oscount#ipcount#ip_app_by_device_cumcount#app_channel_by_ip_countuniq#app_channel_by_os_countuniq,
# next_next_ip_os_device_app_diff#next_prev_ip_os_device_app_diff#next_next_ip_os_device_app_diff_mean#next_next_ip_app_diff#next_prev_ip_app_diff,
# channel_by_app_countuniq#app_oscount#app_os_by_channel_countuniq#app_os_by_device_countuniq#app_os_by_ip_countuniq
#prev,unique,diff,unique2