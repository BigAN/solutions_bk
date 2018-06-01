import pandas as pd

files = [
    "",
]

preds = pd.concat([pd.read_csv(
    '/Users/dongjian/work/meituan/instacart/solutions/meituan_lasthour/stack/regression/stack_data/{}'.format(f)) for f
                   in files], axis=1)

pred = reduce(lambda a, b: a + b, preds) / len(preds)
pred.to_csv('avg-2.csv')

print "Done."
