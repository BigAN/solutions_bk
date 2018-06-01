#cv
py lgbm_cv.py --train "/Users/dongjian/data/meituanKaggleData/train_cv.csv" --test "/Users/dongjian/data/meituanKaggleData/test_cv.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv"  --round 500
#提交最终 out.csv
py lgbm_submit_mt.py --train "/Users/dongjian/data/meituanKaggleData/train_sub.csv" --test "/Users/dongjian/data/meituanKaggleData/test_sub.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv"
#cv 数据生成
py cv_data_producer.py --input /Users/dongjian/data/meituanKaggleData/  --output /Users/dongjian/data/meituanKaggleData/
#所有训练 数据生成
py data_producer.py --input /Users/dongjian/data/meituanKaggleData/  --output /Users/dongjian/data/meituanKaggleData/

#bestset search
py find_bestset.py --train "/Users/dongjian/data/meituanKaggleData/train.csv" --test "/Users/dongjian/data/meituanKaggleData/test.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv"  --round 500 > best_drop_500

#特征选择
py feature_select.py --train "/Users/dongjian/data/meituanKaggleData/train.csv" --test "/Users/dongjian/data/meituanKaggleData/test.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv"  --round 500 > best_drop_500
