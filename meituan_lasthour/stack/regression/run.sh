. ~/.zshrc
export PYTHONPATH=PYTHONPATH:/Users/dongjian/work/meituan/instacart
py execute_base.py --train "/Users/dongjian/data/meituanKaggleData/train_sub.csv" --test "/Users/dongjian/data/meituanKaggleData/test_sub.csv"
py stacking_submit.py