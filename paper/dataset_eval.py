import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import curve_fit


df = pd.read_csv('df_dump.csv')

keys = ["Ground Truth", "Visual Estimation", "Polygon Annotations"]
size = len(df)

for key in keys:
    vals = df[key]
    print(key, vals.mean().round(2), vals.std().round(2), vals.min().round(2), vals.max().round(2), sep="\t")




print()
pairs = (
    ("Visual Estimation", "Ground Truth"),
    ("Polygon Annotations", "Ground Truth"),
    ("Polygon Annotations", "Visual Estimation"),
    # ("Polygon Annotations (side A)", "Polygon Annotations (side B)"),
)
for key1, key2 in pairs:
        vals1 = df[key1]
        vals2 = df[key2]
        abs_diff = (vals2 - vals1).abs().mean()
        # abs_diff = vals2.mean() - vals1.mean()
        rel_diff = (abs_diff / vals1.mean()) * 100
        corr = vals1.corr(vals2)
        max_dif = (vals2 - vals1).abs().max()
        print(key1, key2,  f"{abs_diff.round(2)} ({rel_diff.round(2)} %)", max_dif.round(2), corr.round(2), sep="\t")


df = df.sort_values(by=["Polygon Annotations"])
difference =  df["Polygon Annotations (side A)"] - df["Polygon Annotations (side B)"]
[a, b], _ = curve_fit(lambda x, a, b: a * x + b, range(len(df)), difference)
plt.figure()

plt.scatter(range(len(df)), difference, label="Difference between sides A and B")
plt.axline((0, b), slope=a, color='C0', label=f'Trendline (y = {round(a,2)} * x + {round(b,2)})')
plt.ylabel("# of Flowers")
plt.xlabel("Index of a Sorted Tree")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()



