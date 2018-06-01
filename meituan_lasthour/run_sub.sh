. ~/.zshrc
echo "########### sub: "$1"###########" >>sub_rs

py data_producer.py --input /Users/dongjian/data/meituanKaggleData/  --output /Users/dongjian/data/meituanKaggleData/ >> sub_rs
py lgbm_submit_mt.py --train "/Users/dongjian/data/meituanKaggleData/train_sub.csv" --test "/Users/dongjian/data/meituanKaggleData/test_sub.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv" --round 1000 >> sub_rs
git commit -am "sub:$1"