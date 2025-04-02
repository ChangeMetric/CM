import pandas as pd
import numpy as np
from info_gain import info_gain
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,roc_auc_score

def calculate_gain_ratio(dataset_name):
    f=open("gain_ratio_b.csv","w",encoding="utf-8")
    features_A = pd.read_csv(f"../data2/{dataset_name}/fine_tune_runtime.csv")
    features_B = pd.read_csv(f"../data2/{dataset_name}/diff_features.csv")
    for i in features_B.columns:
        features_B.rename(columns={i: 'diff_'+i}, inplace=True)
    features_C = pd.read_csv(f"../data2/{dataset_name}/diff_features2.csv")
    label = pd.read_csv(f"../data2/{dataset_name}/label.csv")['label']
    features_ABC = pd.concat([features_A,features_B, features_C],join='outer',axis=1) 
    res = []
    num=1
    for columns in features_B.columns:
        print(num, columns)
        num+=1
        res.append([info_gain.info_gain_ratio(label, features_B[columns]), columns])
    res.sort(key=lambda x:x[0])
    for i in res:
        f.write(f"{i[1]},{i[0]}\n")
        print(f"{i[0]}\t{i[1]}")

def paint_bar():
    gain_ratio = pd.read_csv(f"../results/gain_ratio/gain_ratio.csv")
    gain_ratio = gain_ratio.fillna(0)
    min_value = gain_ratio['gain_ratio'].min()
    max_value = gain_ratio['gain_ratio'].max()
    gain_ratio['gain_ratio_normalized'] = (gain_ratio['gain_ratio'] - min_value) / (max_value - min_value)
    print(gain_ratio['gain_ratio_normalized'])
    sns.set_theme(style="darkgrid")
    sns.barplot(x="metric", y="gain_ratio_normalized", data=gain_ratio[:50])
    plt.ylim(0.92, 1.0)
    plt.xticks(rotation=90,fontsize=8)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

if __name__ =="__main__":
    # calculate_gain_ratio('all')
    paint_bar()