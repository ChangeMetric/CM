import pandas as pd
import numpy as np
from matplotlib.cbook import boxplot_stats
import matplotlib.pyplot as plt
import seaborn as sns

datasets = ['blob','circle','mnist','cifar10','reuters','imdb']
model_num = {'blob':39,'circle':36, 'cifar10':35, 'mnist':78, 'reuters':32,'imdb':13}

def main():
    all_data = []
    for idx, dataset_name in enumerate(datasets):
        df = pd.read_csv(f"../metrics/{dataset_name}/diff_features2.csv")
        column = 'val_acc_euclidean_distance'
        nonbuggy_metric_value = []  
        le_metric_value = []    
        for i in range(model_num[dataset_name]):
            for j in range(5):
                nonbuggy_metric_value.append(df[column][i*30+j])
                le_metric_value.append(df[column][i*30+j+10])
                all_data.append({'Dataset': dataset_name, 'Type': 'Non-Faulty', 'Metric': df[column][i*30+j]})
                all_data.append({'Dataset': dataset_name, 'Type': 'Faulty', 'Metric': df[column][i*30+j+10]})
        
    plot_df = pd.DataFrame(all_data)
    plt.figure(figsize=(12,6))
    ax = sns.boxplot(
        data=plot_df,
        x='Dataset',
        y='Metric',
        hue='Type',
        width=0.3,
        showfliers=False
    )

    ax.set_xticklabels(['Blob', 'Circle', 'MNIST', 'CIFAR-10', 'Reuters', 'IMDb'])

    plt.ylabel('Change Metric(val_acc_euclidean_distance)')
    plt.legend(title='')
    plt.show()

def dict_to_df(d, tag):
    rows = []
    for k, v in d.items():
        arr = np.array(v, dtype=float)
        arr = arr[np.isfinite(arr)]

        for x in arr:
            rows.append({
                "dataset": k,
                "value": x,
                "group": tag
            })
    return pd.DataFrame(rows)


def main2():
    metric_dsdm = ['diff_val_acc_mean','diff_decrease_acc_mean','diff_decrease_acc_std','diff_decrease_acc_skew','diff_decrease_acc_var',
                   'diff_decrease_acc_sem','diff_nan_acc_mean','diff_nan_acc_std','diff_nan_acc_skew','diff_nan_acc_median','diff_nan_acc_var',
                   'diff_nan_acc_sem','diff_nan_acc_max','diff_nan_acc_min','diff_val_acc_median','diff_acc_median','diff_acc_var','diff_acc_sem',
                   'diff_acc_std','diff_acc_skew','diff_acc_max','diff_acc_mean']
    metric_mdcm = ['decrease_acc_ks_p','nan_acc_mmd_rbf','val_acc_mmd_rbf','val_acc_euclidean_distance','val_acc_manhatttan_distance',
                   'decrease_acc_euclidean_distance','decrease_acc_manhatttan_distance','decrease_acc_mmd_rbf','nan_acc_euclidean_distance',
                   'nan_acc_manhatttan_distance','acc_cosine_similarity','val_acc_cosine_similarity','acc_manhatttan_distance','acc_euclidean_distance',
                   'acc_mmd_rbf']
    res_range_ratio = {'blob':[],'circle':[], 'cifar10':[], 'mnist':[], 'reuters':[],'imdb':[]}
    res_iqr_ratio = {'blob':[],'circle':[], 'cifar10':[], 'mnist':[], 'reuters':[],'imdb':[]}
    for idx, dataset_name in enumerate(datasets):
        for metric in metric_dsdm:
            df = pd.read_csv(f"../metrics/{dataset_name}/diff_features.csv")
            metric = metric.replace("diff_","")
            nonbuggy_metric_value = []  
            le_metric_value = []       
            for i in range(model_num[dataset_name]):
                for j in range(5):
                    nonbuggy_metric_value.append(df[metric][i*30+j])
                    le_metric_value.append(df[metric][i*30+j+10])

            stats_nonbuggy = boxplot_stats(nonbuggy_metric_value)[0]
            stats_le = boxplot_stats(le_metric_value)[0]

            range_nonbuggy = stats_nonbuggy['whishi'] - stats_nonbuggy['whislo']
            range_buggy = stats_le['whishi'] - stats_le['whislo']
            ratio_range =  range_buggy / range_nonbuggy if range_nonbuggy != 0 else float('inf')
            res_range_ratio[dataset_name].append(ratio_range)

            iqr_nonbuggy = stats_nonbuggy['q3'] - stats_nonbuggy['q1']
            iqr_buggy = stats_le['q3'] - stats_le['q1']
            ratio_iqr = iqr_buggy / iqr_nonbuggy if iqr_nonbuggy != 0 else float('inf')
            res_iqr_ratio[dataset_name].append(ratio_iqr)
        
        for metric in metric_mdcm:
            df = pd.read_csv(f"../metrics/{dataset_name}/diff_features2.csv")
            nonbuggy_metric_value = [] 
            le_metric_value = []      
            for i in range(model_num[dataset_name]):
                for j in range(5):
                    nonbuggy_metric_value.append(df[metric][i*30+j])
                    le_metric_value.append(df[metric][i*30+j+10])

            stats_nonbuggy = boxplot_stats(nonbuggy_metric_value)[0]
            stats_le = boxplot_stats(le_metric_value)[0]

            range_nonbuggy = stats_nonbuggy['whishi'] - stats_nonbuggy['whislo']
            range_buggy = stats_le['whishi'] - stats_le['whislo']
            ratio_range =  range_buggy / range_nonbuggy if range_nonbuggy != 0 else float('inf')
            res_range_ratio[dataset_name].append(float(ratio_range))

            iqr_nonbuggy = stats_nonbuggy['q3'] - stats_nonbuggy['q1']
            iqr_buggy = stats_le['q3'] - stats_le['q1']
            ratio_iqr = iqr_buggy / iqr_nonbuggy if iqr_nonbuggy != 0 else float('inf')
            res_iqr_ratio[dataset_name].append(float(ratio_iqr))
    df1 = dict_to_df(res_range_ratio, 'Range Ratio')
    df2 = dict_to_df(res_iqr_ratio, 'IQR Ratio')
    df = pd.concat([df1, df2], ignore_index=True)
    
    plt.figure(figsize=(5,5))
    g = sns.displot(
        data=df,
        x="value",
        hue="group",
        col="dataset",      
        kind="ecdf",
        col_wrap=3,         
        height=3,
        facet_kws=dict(sharex=False, sharey=False)
    )
    
    for ax in g.axes.flat:
        ax.set_xscale("log")  

    custom_titles = ["Blob", "Circle", "CIFAR-10", "MNIST", "Reuters", "IMDb"]
    for ax, title in zip(g.axes.flat, custom_titles):
        ax.set_title(title, fontsize=12)

    g._legend.remove()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    main2()