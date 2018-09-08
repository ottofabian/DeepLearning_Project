import json
from pprint import pprint
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

# json = np.array(json.loads("./resultPixelLevelSemanticLabeling.json")["confMatrix"])
with open("./resultPixelLevelSemanticLabeling.json") as data_file:
    js = json.load(data_file)
    conf = np.array(js["confMatrix"], dtype=np.int32)
    label = js["labels"]

label = np.array(sorted(label, key=label.get))

idx = (conf != 0).any(axis=0)
conf = conf[:, idx][idx, :]
sum_row = np.sum(conf, axis=0)
conf = conf / sum_row

df_cm = pd.DataFrame(conf, index=[i for i in label[idx]],
                     columns=[i for i in label[idx]])
df_cm.fillna(0)
plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix")
ax = sn.heatmap(df_cm, annot=False)
ax.set_xlabel("Prediction")
ax.set_ylabel("True")
plt.show()
