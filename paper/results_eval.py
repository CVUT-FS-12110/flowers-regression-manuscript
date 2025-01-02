import os
import glob
import re

import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

df = pd.read_csv('df_dump.csv', index_col="orig_name")

reference_method_yolo_path = "../yolo/results"
regeression_paths = [ # TODO put correct folders with results when finish your experiments
    ("result-fpn", "../multiview/results/FPN"),
    ("result-standard", "../multiview/results/standard"),
    ("result-fpn-exponential", "../multiview/results/FPN exp"),
    ("result-countnet", "../multiview/results/Countnet")
]

for result_label, result_path in regeression_paths:
    for ground_truth in [True, False]:
        filepaths = os.path.join(result_path, f"results_*_{ground_truth}.tsv")
        for filepath in glob.glob(filepaths):
            result_df = pd.read_csv(filepath, sep="\t", index_col=0, header=None,
                                    names=["Index", "Prediction", "Error", "Epoch"]).reindex(df.index)
            col_name = f"{result_label}-{ground_truth}-" + filepath.split("_")[-2]
            assert result_df["Error"].index.equals(df.index), "Wrong!"
            df[col_name] = result_df["Error"]

df = df.copy() # defragmentation for better performance

# get and fit YOLO data from TSV files
filepaths = os.path.join(reference_method_yolo_path, f"result_*.tsv")
for filepath in glob.glob(filepaths):
    result_df = pd.read_csv(filepath, sep="\t", header=None, names=["File", "Prediction"])
    result_df.index = [re.search(r"/(noc_\d+_\d+|den_\d+_\d+)", text).group(1) for text in result_df["File"] if text]
    # print(result_df.head()) # TODO check out again
    result_series = result_df.groupby(result_df.index)['Prediction'].sum()
    assert df.index.equals(result_series.index), "Wrong!"
    for indicator, name in ((True, "Ground Truth"), (False, "Visual Estimation")):
        [a, b], _ = curve_fit(lambda x, a, b: a * x + b, result_series, df['Ground Truth'])
        fitted_result_series = result_series * a + b
        col_name = f"result-yolo-{indicator}-" + filepath.split("_")[-1].split(".")[0]
        df[col_name] = df[name] - fitted_result_series
        assert df[col_name].abs().sum() < (df[name] - result_series).abs().sum(), "Wrong!"


df = df.copy() # defragmentation for better performance

NAMES = {
    "result-fpn": "Proposed method",
    "result-countnet": "CountNet (Reference 1)",
    "result-yolo": "YOLO (Reference 2)",
}
for truth in (False, True):
    truth_key = "Ground Truth" if truth else "Visual Estimation"
    print(truth_key)
    for name in NAMES.keys():
        print(NAMES[name], end="\t")
        keys = [key for key in df.keys() if f"{name}-{truth}" in key]
        mae = df[keys].abs().mean(axis=0).mean(axis=0)
        rpe = (mae / df[truth_key].mean()) * 100
        rmse = (df[keys] ** 2).mean(axis=0).mean(axis=0) ** 0.5
        corrs = df[keys].add(df[truth_key], axis=0).corrwith(df[truth_key]).mean()
        std = df[keys].abs().mean(axis=0).std()
        print(f"{mae.round(2)} $\pm$ {std.round(2)}", rmse.round(2), f"{rpe.round(2)} %", corrs.round(3), sep="\t", end="\t")
        print()
    print()


print()
print()
print()

NAMES = {
    "result-fpn": "ResNet + FPN + FC",
    "result-standard": "ResNet + FC",
    "result-fpn-exponential": "ResNet + FPN + bad learning",
}
for truth in (False, True):
    truth_key = "Ground Truth" if truth else "Visual Estimation"
    print(truth_key)
    for name in NAMES.keys():
        print(NAMES[name], end="\t")
        keys = [key for key in df.keys() if f"{name}-{truth}" in key]
        mae = df[keys].abs().mean(axis=0).mean(axis=0)
        rpe = (mae / df[truth_key].mean()) * 100
        rmse = (df[keys] ** 2).mean(axis=0).mean(axis=0) ** 0.5
        corrs = df[keys].add(df[truth_key], axis=0).corrwith(df[truth_key]).mean()
        std = df[keys].abs().mean(axis=0).std()
        print(f"{mae.round(2)} $\pm$ {std.round(2)}", rmse.round(2), f"{rpe.round(2)} %", corrs.round(3), sep="\t", end="\t")
        print()
    print()

