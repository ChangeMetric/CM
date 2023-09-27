import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from  matplotlib import cm
import matplotlib.patches as mpatches

from sklearn.metrics import accuracy_score, classification_report
import argparse


dataset=['blob','circle','mnist','cifar10','reuters','imdb']
# dataset=['all']
datasets=['Blob','Circle','MNIST','CIFAR-10','Reuters','IMDB']

def get_features():
    features_all = 0
    for dataset_name in dataset:
        features_diff = pd.read_csv(f"Evaluation/{dataset_name}/features_diff.csv")
        # features_diff = data.drop(axis=1,columns=['model','mutant','label'])
        # features_diff = np.array(features_diff)
        for i in features_diff.columns:
            if i not in ['model','mutant','label', 'iter']:
                features_diff.rename(columns={i: 'diff_'+i}, inplace=True)
        features_diff.insert(0, 'dataset', dataset_name, allow_duplicates = False)
        features_diff.rename(columns={'model':'model_name'},inplace=True)
        
        features = pd.read_csv(f"Evaluation/{dataset_name}/features_old.csv")
        features.rename(columns={'model':'model_name'},inplace=True)
        merged_features = pd.merge(features_diff, features, on=['model_name','mutant','label','iter'])
                   
        features_change = pd.read_csv(f"Evaluation/{dataset_name}/change_features.csv")
        features_change = features_change.drop(axis=1,columns=['dataset_name'])          
        merged_features = pd.merge(merged_features, features_change, on=['model_name','mutant','iter'])

        if isinstance(features_all,int):
            features_all = merged_features
        else:
            features_all = pd.concat([features_all,merged_features],join='outer',axis=0)
        # print(merged_features.columns)
        # features_all.to_csv('all_features.csv',index=False)
        # print(features_diff.isnull().values.any())
        # print(features.isnull().values.any())
        # print(features_change.isnull().values.any())
        # exit()
    return features_all


# 计算每个类别的f1 
def paint_boxplot(d):
    tmp_dict={1:[],2:[],3:[],4:[],5:[]}
    for key,value in d.items():
        for label_num in range(1,6):
            tmp_dict[label_num].append(value[str(label_num)])
    
    fig, axes = plt.subplots(1,5)
    # plt.tight_layout()
    fig.set_size_inches(10,3.3)
    font = FontProperties(fname=r"C:\Windows\Fonts\arial.ttf", size=10)
    plt.subplots_adjust(left=0.075, bottom=0.39, right=0.98, top=0.9, wspace=0.5, hspace=0.5)
    classifier=['DT','RF','NB','KNN','LR']
    num=0
    for key,value in d.items():
        # plt.subplot(1,5,num+1)
        box1 = axes[num].boxplot([value for key,value in d[key].items()],patch_artist=True,
            boxprops={"facecolor": f"C0",
                        "edgecolor": "black",
                        "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            # meanprops={'marker':'+',
            #             'markerfacecolor':'k',
            #             'markeredgecolor':'k',
            #             'markersize':5},
            flierprops=dict(marker='o', markersize=4))
        axes[num].grid( ls='--', lw=0.25, alpha=0.8)
        axes[num].set_xticks([1,2,3,4,5])
        # 设置x轴刻度标签,并旋转90°
        # axes[num].set_xticklabels(['Noise Perturbation','Label Error','Data Repetition','Data Missing','Data Shuffle'],rotation=75,FontProperties=font)
        axes[num].set_xticklabels(['NP','LE','DR','DM','DS'],rotation=0,FontProperties=font)

        y_labels = axes[num].get_yticklabels()
        for label in y_labels:
            label.set_fontproperties(font)
            label.set_fontsize(10)  # 设置字体大小，可以根据需要调整
        axes[num].set_title(classifier[num],FontProperties=font)
        if num == 0:
            axes[num].set_ylabel('F1-score',FontProperties=font)
        num+=1
    plt.show()

# dis3
# 计算每个类别的f1
def evaluation_each_class(dataset):
    res_all={}
    for dataset_name in dataset:
        print(f"{dataset_name}")
        res_all[dataset_name] = {}

        for clf in ['dt_pred','rf_pred','nb_pred','knn_pred','lr_pred']:
            res_tmp={'1':[],'2':[],'3':[],'4':[],'5':[]}
            for i in range(10):
                for j in range(10):
                    data = pd.read_csv(f'Evaluation_sp/results/{dataset_name}/BC/{i}_{j}.csv')
                    t = classification_report(data['y_test'], data[clf],output_dict=True)
                    # print(t['0']['f1-score'])
                    for label_num in range(1,6):
                        if str(label_num) in t:
                            res_tmp[str(label_num)].append(t[str(label_num)]['f1-score'])
            res_all[dataset_name][clf] = res_tmp
        paint_boxplot(res_all[dataset_name])

def paint_bar(d):
    categories = ['NP','LE','DR','DM','DS']
    # 创建一个图形和一个轴对象
    fig, axes = plt.subplots(1,7)
    # plt.tight_layout()
    fig.set_size_inches(10,3.3)
    plt.subplots_adjust(left=0.05, bottom=0.2, right=1, top=0.9, wspace=0.5, hspace=0.5)
    font = FontProperties(fname=r"C:\Windows\Fonts\arial.ttf", size=10)
    
    num=0
    for key, value in d.items():
        # 绘制堆叠条形图
        # bars = []
        # bottom = [0] * len(categories)
        # for i, percentage in enumerate(value):
        #     bar = ax.bar(categories, percentage, bottom=bottom, label=f'Part {i+1}')
        #     bars.append(bar)
        #     bottom = [sum(x) for x in zip(bottom, percentage)]
        axes[num].bar(categories, value[0], label= 'Faulty', color='C1')
        axes[num].bar(categories, value[1], bottom=value[0], label= 'Correct',color='C0')
        
        # 添加图例
        # axes[num].legend()

        axes[num].set_ylim(0, 1)
        axes[num].set_xticklabels(categories,rotation=90,FontProperties=font)
        y_labels = axes[num].get_yticklabels()
        for label in y_labels:
            label.set_fontproperties(font)
            label.set_fontsize(10)  # 设置字体大小，可以根据需要调整
        # 设置图形标题和标签
        axes[num].set_title(datasets[num], fontsize=10)
        num+=1

    axes[0].set_ylabel('Propotion',FontProperties=font)

    legend_part1 = mpatches.Patch(color='C1', label='Faulty')
    legend_part2 = mpatches.Patch(color='C0', label='Correct')
    custom_legend = plt.legend(handles=[legend_part1, legend_part2], loc='center right')
    
    axes[6].axis('off')
    axes[6].add_artist(custom_legend)

    # axes[6].legend(['Faulty','skyblue'], loc='center right',fontsize='small')    

    plt.show()

# dis4
# 计算标签分布
def draw_label_cat():
    all_label = [1950, 1800, 3900, 1750, 640, 260]
    num=0
    final_res = {}
    for dataset_name in dataset:
        data = pd.read_csv(f"Evaluation_sp/{dataset_name}/features_diff.csv")
        res = [0,0,0,0,0,0]
        for i in data['label']:
            res[int(i)]+=1
        res = res[1:]
        res = [i/all_label[num] for i in res]
        # for i in res:
        #     draw_res.append([i/all_label[num], 1-i/all_label[num]])
        res_2 = []
        for i in res:
            res_2.append(1-i)
        # print(draw_res)
        num+=1
        final_res[dataset_name] = [res, res_2]

    # print(final_res)
    paint_bar(final_res)


# dis2
# gain ratio 比例
def paint_pie():
    fig, axes = plt.subplots(1,3)
    labels=['DF','DSDM','MDCM']
    
    colors = cm.Paired(np.arange(3)/3) # colormaps: Paired, autumn, rainbow, gray,spring,Darks
    # 重新设置字体大小
    proptease = FontProperties(size=8)
    # proptease.set_size(8)
    # font size include: ‘xx-small’,x-small’,'small’,'medium’,‘large’,‘x-large’,‘xx-large’ or number, e.g. '12'

    # colors=['C0']
    patches0, texts0, autotexts0 = axes[0].pie([17,16,17],autopct='%1.2f%%',shadow=False, startangle=110)
    patches1, texts1, autotexts1 = axes[1].pie([20,20,10],autopct='%1.2f%%',shadow=False, startangle=110)
    axes[0].axis('equal')
    axes[1].axis('equal')
    
    axes[0].set_title('Fault Detection',fontsize='medium')
    axes[1].set_title('Fault Diagnosis',fontsize=10)

    axes[2].axis('off')
    axes[2].legend(patches0, labels, loc='center right',fontsize='small')

    plt.setp(autotexts0, fontproperties=proptease)
    plt.setp(texts0, fontproperties=proptease)
    plt.setp(autotexts1, fontproperties=proptease)
    plt.setp(texts1, fontproperties=proptease)

    # axes[2].axis('off')
    # axes[2].legend(patches0, labels, loc='center left',fontproperties=proptease)


    plt.show()

# dis1
classifier = ['DT','RF','NB','KNN','LR']
def filter_data(df, metric):
    if metric == 'f1':
        offset = 0
    elif metric == 'acc':
        offset = 1
    elif metric == 'roc':
        offset = 2
    temp_list = []
    for i in range(100):
        temp_list.append(df.iloc[offset+i*3])
    return pd.DataFrame(temp_list)

def pca_compare(dataset):
    fig, axes = plt.subplots(1,4)
    fig.set_size_inches(10,3.6)
    plt.subplots_adjust(left=0.075, bottom=0.1, right=0.95, top=0.9, wspace=0.5, hspace=0.5)
    font = FontProperties(fname=r"C:\Windows\Fonts\arial.ttf", size=9)
    for dataset_name in dataset:
        num = 0
        for m in ['f1','acc','roc']:
            res = []
            res_pca = []
            for classifier_name in classifier:
                df = pd.read_csv(f"Evaluation/results/{dataset_name}_{classifier_name}.csv")
                df_pca = pd.read_csv(f"Evaluation/results_PCA/{dataset_name}_{classifier_name}.csv")
                new_df = filter_data(df, m)
                new_df_pca = filter_data(df_pca, m)
                res.append(new_df['BC'])
                res_pca.append(new_df_pca['BC'])


            box1 = axes[num].boxplot(res,patch_artist=True,showmeans=True,positions=[1,4,7,10,13],
            boxprops={"facecolor": f"C0",
                        "edgecolor": "black",
                        "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                        'markerfacecolor':'k',
                        'markeredgecolor':'k',
                        'markersize':5},
            flierprops=dict(marker='o', markersize=4))

            box2 = axes[num].boxplot(res_pca,positions=[2,5,8,11,14],patch_artist=True,showmeans=True,
                    boxprops={"facecolor": "C1",
                            "edgecolor": "black",
                            "linewidth": 0.5},
                    medianprops={"color": "k", "linewidth": 0.5},
                    meanprops={'marker':'+',
                            'markerfacecolor':'k',
                            'markeredgecolor':'k',
                            'markersize':5},
                    flierprops=dict(marker='o', markersize=2))

            axes[num].grid( ls='--', lw=0.25, alpha=0.8)

            axes[num].set_xticks([1.5,4.5,7.5,10.5,13.5])
            axes[num].set_xticklabels(classifier,FontProperties=font)
            y_labels = axes[num].get_yticklabels()
            for label in y_labels:
                label.set_fontproperties(font)
                label.set_fontsize(10)  # 设置字体大小，可以根据需要调整
            if num == 0:
                axes[num].set_ylabel('F1-score',FontProperties=font)
            if num == 1:
                axes[num].set_ylabel('Accuracy',FontProperties=font)
            if num == 2:
                axes[num].set_ylabel('AUC',FontProperties=font)        
            # axes[num].legend(handles=[box1['boxes'][0],box2['boxes'][0]],labels=['Non-buggy','Faulty'],loc='upper right')   
            num+=1

       
    legend_part1 = mpatches.Patch(color='C0', label='Without PCA')
    legend_part2 = mpatches.Patch(color='C1', label='With PCA')
    custom_legend = plt.legend(handles=[legend_part1, legend_part2], loc='center left')
    axes[3].axis('off')
    axes[3].add_artist(custom_legend)   
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', choices=['1','2','3','4'], help='Choose the discussion')
    args = parser.parse_args()
   
    if args.d == "1":
        pca_compare(['all'])
    if args.d == "2":
        paint_pie()
    if args.d == "3":
        evaluation_each_class(['all'])
    if args.d == "4":
        draw_label_cat()

    
