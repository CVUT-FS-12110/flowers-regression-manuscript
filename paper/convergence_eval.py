import os
import glob

import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

# proposed_method_path = "../multiview/results/3-lr_01-xavier_01-dropout_cossine-decay"
# proposed_method_path = "../multiview/results/CountNet (cvdev)"
# proposed_method_path = "../multiview/results/cross_attention V2"
# proposed_method_path = "../multiview/results/standard LR modified"
proposed_method_path = "../multiview/results/standard 300epoch (cvdev)"

all_dataframes = []

# Iterate through all files matching the pattern
for ground_truth in [False, ]:
    filepaths = os.path.join(proposed_method_path, f"convergence_*_{ground_truth}.tsv")
    for filepath in glob.glob(filepaths):
        convergence_df = pd.read_csv(filepath, sep="\t", header=None)
        all_dataframes.append(convergence_df)

merged_df = pd.concat(all_dataframes, axis=1)
average_series = merged_df.mean(axis=1)

merged_df.mean(axis=1).plot(logy=True)
plt.show()