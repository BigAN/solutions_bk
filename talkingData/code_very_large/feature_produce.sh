python -u 9778_data_produce.py --debug f --subsample f  --nchunk 60000000 --val 4000000 >> basicfea
python -u 9778_data_produce__prev_click.py --debug f --subsample f   --nchunk 60000000 --val 4000000  --fname prev >> prev
python -u 9778_data_produce__diff_feature.py --debug f --subsample f  --nchunk 60000000 --val 4000000 --fname diff >> diff
python -u 9778_data_produce__more_unique.py --debug f --subsample f  --nchunk 60000000 --val 4000000 --fname unique >> unique
python -u 9778_data_produce__more2_unique.py --debug f --subsample f  --nchunk 60000000 --val 4000000 --fname unique2 >> unique2
#python -u 9778_data_produce__rolling.py --debug t --subsample f  --nchunk 60000000 --val 4000000 --fname rolling >> rolling
