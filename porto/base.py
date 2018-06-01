import os
import pandas as pd

submissions_path = "/Users/dongjian/work/meituan/instacart/solutions/porto/input/"
all_files = os.listdir(submissions_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(submissions_path, f), index_col=0)\
        for f in all_files]
concat_df = pd.concat(outs, axis=1)
cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

# Apply ranking, normalization and averaging
concat_df["target"] = (concat_df.rank() / concat_df.shape[0]).mean(axis=1)
concat_df.drop(cols, axis=1, inplace=True)

# Write the output
concat_df.to_csv("./kagglemix.csv")
