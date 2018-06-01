. ~/.zshrc
py cv_data_producer.py --input /Users/dongjian/data/meituanKaggleData/  --output /Users/dongjian/data/meituanKaggleData/
py feature_select.py --train "/Users/dongjian/data/meituanKaggleData/train_cv.csv" --test "/Users/dongjian/data/meituanKaggleData/test_cv.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv"  --round 500 > fel_sel
py find_bestset.py --train "/Users/dongjian/data/meituanKaggleData/train_cv.csv" --test "/Users/dongjian/data/meituanKaggleData/test_cv.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv"  --round 500  > best_set