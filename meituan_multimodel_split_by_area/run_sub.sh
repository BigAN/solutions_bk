. ~/.zshrc
echo "########### sub: "$1"###########"
py data_producer.py --input /Users/dongjian/data/meituanKaggleData/  --output /Users/dongjian/data/meituanKaggleData/ >> rs
py lgbm_submit_mt.py --train "/Users/dongjian/data/meituanKaggleData/train_sub.csv" --test "/Users/dongjian/data/meituanKaggleData/test_sub.csv" --output "/Users/dongjian/data/meituanKaggleData/out.csv" --round 800 >> rs
git commit -am "sub:$1"