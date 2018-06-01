. ~/.zshrc
echo "###########"$1"###########" >>rs
py cv_data_producer.py --input /Users/dongjian/data/meituanKaggleData/  --output /Users/dongjian/data/meituanKaggleData/ >> rs
py lgbm_cv.py --train "/Users/dongjian/data/meituanKaggleData/train_cv.csv" --test "/Users/dongjian/data/meituanKaggleData/test_cv.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv"  --round 1600 --label delivery_duration >> rs
git commit -am "$1"