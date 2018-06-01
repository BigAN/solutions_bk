. ~/.zshrc
py cv_data_producer_test.py --input /Users/dongjian/data/meituanKaggleData/  --output /Users/dongjian/data/meituanKaggleData/
py lgbm_cv.py --train "/Users/dongjian/data/meituanKaggleData/train_cv.csv" --test "/Users/dongjian/data/meituanKaggleData/test_cv.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv"  --round 500 --label delivery_duration