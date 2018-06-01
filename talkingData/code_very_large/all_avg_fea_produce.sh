nohup python -u 9778_data_produce.py --debug f --subsample f  --nchunk 179903890 --val 4000000 2>&1 >> basicfea &
nohup python -u 9778_data_produce__prev_click.py --debug f --subsample f   --nchunk 179903890 --val 4000000 2>&1  --fname prev >> prev &
nohup python -u 9778_data_produce__diff_feature.py --debug f --subsample f  --nchunk 179903890 --val 4000000 2>&1 --fname diff >> diff &
nohup python -u 9778_data_produce__more_unique.py --debug f --subsample f  --nchunk 179903890 --val 4000000 2>&1 --fname unique >> unique &
nohup python -u 9778_data_produce__more2_unique.py --debug f --subsample f  --nchunk 179903890 --val 4000000 2>&1 --fname unique2 >> unique2 &

