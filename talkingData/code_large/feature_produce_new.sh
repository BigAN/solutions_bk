python -u 9778_data_produce.py --debug f --subsample f --frm 184903889  --nchunk 184903889 --val 25000000 >> basicfea
python -u 9778_data_produce__lag_feature.py --debug f --subsample f --frm 184903889  --nchunk 184903889 --val 25000000 >> basicfea

#python -u 9778_data_produce__lag_feature.py --debug f --subsample f  --nchunk 25000000 --val 2500000 --fname lagfea >> lagfea
#python -u 9778_data_produce__prev_click.py --debug f --subsample f  --nchunk 25000000 --val 2500000 --fname prev >> prev
#python -u 9778_data_produce__diff_feature.py --debug f --subsample f  --nchunk 25000000 --val 2500000 --fname diff >> diff
