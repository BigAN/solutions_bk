. ~/.zshrc
echo "###########"$1"###########" >>rs
py data_producer.py --input /Users/dongjian/data/meituanKaggleData/  --output /Users/dongjian/data/meituanKaggleData/ >> rs
py lgbm_cv_5f.py --train "/Users/dongjian/data/meituanKaggleData/train_sub.csv" --test "/Users/dongjian/data/meituanKaggleData/test_cv.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv"  --round 500 --label delivery_duration >> rs
git commit -am "$1"

