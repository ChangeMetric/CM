from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, f1_score, accuracy_score
import pandas as pd
import numpy as np
import os
from cliffs_delta import cliffs_delta
import scipy.stats as stats
import time
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import argparse

classifiers = ['DT','RF','NB','KNN','LR']
class Classifier:
    def __init__(self, features, label) -> None:
        self.label = label
        self.features = features

        self.classifiers = {}
        self.pred = {}
        self.classification_reports = {}
    
    def my_split(self, seed):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.label, test_size=0.3, random_state=seed, stratify=self.label)

    def train_pred(self, classifier):
        if classifier == 'DT':
            clf = DecisionTreeClassifier(criterion='entropy', random_state=10)
            self.classifiers['DT'] = clf.fit(self.X_train, self.y_train)  
            self.pred['DT'] = clf.predict(self.X_test)  
        elif classifier == 'RF':
            clf = RandomForestClassifier(n_estimators = 25,criterion='entropy',random_state=0)
            self.classifiers['RF'] = clf.fit(self.X_train, self.y_train)
            self.pred['RF'] = clf.predict(self.X_test)  
        elif classifier == 'NB':
            clf = GaussianNB()
            self.classifiers['NB'] = clf.fit(self.X_train, self.y_train)
            self.pred['NB'] = clf.predict(self.X_test)
        elif classifier == 'KNN':
            clf = KNeighborsClassifier()
            self.classifiers['KNN'] = clf.fit(self.X_train, self.y_train)  
            self.pred['KNN'] = clf.predict(self.X_test)  
        elif classifier == 'LR':
            clf = LogisticRegression(random_state=0,max_iter=10000,solver='liblinear')
            self.classifiers['LR'] = clf.fit(self.X_train, self.y_train)
            self.pred['LR'] = clf.predict(self.X_test)

def train(dataset_name):
    features_A = pd.read_csv(f"../data2/{dataset_name}/fine_tune_runtime.csv")
    features_B = pd.read_csv(f"../data2/{dataset_name}/diff_features.csv")
    features_C = pd.read_csv(f"../data2/{dataset_name}/diff_features2.csv")
    label = pd.read_csv(f"../data2/{dataset_name}/label.csv")['label']

    features_AB = pd.concat([features_A,features_B],join='outer',axis=1) 
    features_AC = pd.concat([features_A,features_C],join='outer',axis=1) 
    features_BC = pd.concat([features_B,features_C],join='outer',axis=1) 
    features_ABC = pd.concat([features_A, features_B,features_C],join='outer',axis=1) 

    s_time=time.time()
    for i in range(10):
        clf_a = Classifier(features_A,label)
        clf_b = Classifier(features_B,label)
        clf_c = Classifier(features_C,label)
        clf_ab = Classifier(features_AB,label)
        clf_ac = Classifier(features_AC,label)
        clf_bc = Classifier(features_BC,label)
        clf_abc = Classifier(features_ABC,label)
        clf_a.my_split(i)
        clf_b.my_split(i)
        clf_c.my_split(i)
        clf_ab.my_split(i)
        clf_ac.my_split(i)
        clf_bc.my_split(i)
        clf_abc.my_split(i)
        for clf_name in classifiers:
            clf_a.train_pred(clf_name)
            clf_b.train_pred(clf_name)
            clf_c.train_pred(clf_name)
            clf_ab.train_pred(clf_name)
            clf_ac.train_pred(clf_name)
            clf_bc.train_pred(clf_name)
            clf_abc.train_pred(clf_name)

            df = pd.DataFrame()
            df['A'] = clf_a.pred[clf_name]
            df['B'] = clf_b.pred[clf_name]
            df['C'] = clf_c.pred[clf_name]
            df['AB'] = clf_ab.pred[clf_name]
            df['AC'] = clf_ac.pred[clf_name]
            df['BC'] = clf_bc.pred[clf_name]
            df['ABC'] = clf_abc.pred[clf_name]
            df['y_test'] = list(clf_bc.y_test)

            if not os.path.exists(f"../results/pred/{dataset_name}"):
                os.makedirs(f"../results/pred/{dataset_name}")
            df.to_csv(f"../results/pred/{dataset_name}/{dataset_name}_pred_{clf_name}_{i}.csv",index=False)

    e_time=time.time()
    with open(f"../results/pred/{dataset_name}/time.txt","w",encoding="utf-8") as f:
        f.write(f"{e_time-s_time}")

metrics = ['A','B','C','AB','AC','BC','ABC']
# metrics = ['A','BC']
def evaluate(dataset_name):
    if not os.path.exists(f"../results/evaluation/"):
        os.makedirs(f"../results/evaluation/")
    res_acc = []
    res_mf1 = []
    res_auc = []
    for clf_name in classifiers:
        # print(clf_name)
        for i in range(10):
            # print(i)
            df = pd.read_csv(f"../results/pred/{dataset_name}/{dataset_name}_pred_{clf_name}_{i}.csv")
            res_acc_item = [f"{clf_name}_{i}"]
            res_mf1_item = [f"{clf_name}_{i}"]
            res_auc_item = [f"{clf_name}_{i}"]
            for m in metrics:
                t = classification_report(df['y_test'],df[m],output_dict=True)
                res_acc_item.append(t['accuracy'])
                res_mf1_item.append(t['macro avg']['f1-score'])
                # res_auc_item.append(t['weighted avg']['f1-score'])
                
                if dataset_name == 'cifar10':
                    y_true = df['y_test'].replace(4,3)
                    y_pred = df[m].replace(4,3)
                    y_true_onehot = np.eye(4)[y_true]
                    y_pred_onehot = np.eye(4)[y_pred]
                elif dataset_name == 'reuters':
                    y_true_onehot = np.eye(5)[df['y_test']]
                    y_pred_onehot = np.eye(5)[df[m]]
                else:
                    y_true_onehot = np.eye(6)[df['y_test']]
                    y_pred_onehot = np.eye(6)[df[m]]
                s = roc_auc_score(y_true_onehot,y_pred_onehot,multi_class='ovr')
                res_auc_item.append(s)
            res_acc.append(res_acc_item)
            res_mf1.append(res_mf1_item)
            res_auc.append(res_auc_item)
    df_acc = pd.DataFrame(res_acc, columns=['index','A','B','C','AB','AC','BC','ABC'])
    df_acc.to_csv(f"../results/evaluation/{dataset_name}_acc.csv",index=False)
    df_mf1 = pd.DataFrame(res_mf1, columns=['index','A','B','C','AB','AC','BC','ABC'])
    df_mf1.to_csv(f"../results/evaluation/{dataset_name}_mf1.csv",index=False)
    df_auc = pd.DataFrame(res_auc, columns=['index','A','B','C','AB','AC','BC','ABC'])
    df_auc.to_csv(f"../results/evaluation/{dataset_name}_auc.csv",index=False)

    # 计算平均值
    # res_acc = []
    # res_mf1 = []
    # res_auc = []
    # for i in range(5):
    #     res_acc_item = [f"{classifiers[i]}"]
    #     res_mf1_item = [f"{classifiers[i]}"]
    #     res_auc_item = [f"{classifiers[i]}"]
    #     for m in ['A','B','C','AB','AC','BC','ABC']:
    #         res_acc_item.append(np.mean(df_acc[m][i*10:(i+1)*10]))
    #         res_mf1_item.append(np.mean(df_mf1[m][i*10:(i+1)*10]))
    #         res_auc_item.append(np.mean(df_auc[m][i*10:(i+1)*10]))
    #     res_acc.append(res_acc_item)
    #     res_mf1.append(res_mf1_item)
    #     res_auc.append(res_auc_item)
    # df_acc = pd.DataFrame(res_acc, columns=['index','A','B','C','AB','AC','BC','ABC'])
    # df_acc.to_csv(f"../results/evaluation/{dataset_name}_acc_avg.csv",index=False)
    # df_mf1 = pd.DataFrame(res_mf1, columns=['index','A','B','C','AB','AC','BC','ABC'])
    # df_mf1.to_csv(f"../results/evaluation/{dataset_name}_mf1_avg.csv",index=False)
    # df_auc = pd.DataFrame(res_auc, columns=['index','A','B','C','AB','AC','BC','ABC'])
    # df_auc.to_csv(f"../results/evaluation/{dataset_name}_auc_avg.csv",index=False)

def wtl(pvalue, d):
    if pvalue < 0.05 and d >=0.147:
        return 'win'
    elif pvalue < 0.05 and d <=-0.147:
        return 'lose'
    else:
        return 'tie'    

# 'blob','circle','mnist',,'reuters','imdb','all'
datasets=['blob','circle','mnist','cifar10','reuters','imdb','all']
# datasets=['blob','circle']
def hypothesis_testing(one, two):
    acc = {'win': 0 , 'tie': 0, 'lose': 0}
    mf1 = {'win': 0 , 'tie': 0, 'lose': 0}
    auc = {'win': 0 , 'tie': 0, 'lose': 0}
    for i in range(5):
        for dataset_name in datasets:
            df_acc = pd.read_csv(f"../results/evaluation/{dataset_name}_acc.csv")
            df_mf1 = pd.read_csv(f"../results/evaluation/{dataset_name}_mf1.csv")
            df_auc = pd.read_csv(f"../results/evaluation/{dataset_name}_auc.csv")
            p_acc = stats.wilcoxon(df_acc[one][i*10:(i+1)*10], df_acc[two][i*10:(i+1)*10]).pvalue
            d_acc, res_acc = cliffs_delta(df_acc[one][i*10:(i+1)*10], df_acc[two][i*10:(i+1)*10])

            p_mf1 = stats.wilcoxon(df_mf1[one][i*10:(i+1)*10], df_mf1[two][i*10:(i+1)*10]).pvalue
            d_mf1, res_mf1 = cliffs_delta(df_mf1[one][i*10:(i+1)*10], df_mf1[two][i*10:(i+1)*10])

            p_auc = stats.wilcoxon(df_auc[one][i*10:(i+1)*10], df_auc[two][i*10:(i+1)*10]).pvalue
            d_auc, res_auc = cliffs_delta(df_auc[one][i*10:(i+1)*10], df_auc[two][i*10:(i+1)*10])

            acc[wtl(p_acc, d_acc)]+=1
            mf1[wtl(p_mf1, d_mf1)]+=1
            auc[wtl(p_auc, d_auc)]+=1

            # print(f"{classifiers[i]}\t{dataset_name}\t",end="")
            print(f"{np.mean(df_acc[one][i*10:(i+1)*10])}\t{np.mean(df_acc[two][i*10:(i+1)*10])}\t{p_acc}\t{d_acc}\t{wtl(p_acc, d_acc)}\t",end="")
            print(f"{np.mean(df_mf1[one][i*10:(i+1)*10])}\t{np.mean(df_mf1[two][i*10:(i+1)*10])}\t{p_mf1}\t{d_mf1}\t{wtl(p_mf1, d_mf1)}\t",end="")
            print(f"{np.mean(df_auc[one][i*10:(i+1)*10])}\t{np.mean(df_auc[two][i*10:(i+1)*10])}\t{p_auc}\t{d_auc}\t{wtl(p_auc, d_auc)}")
        # print("\n")
            # print(classifiers[i], dataset_name, p_acc, d_acc, wtl(p_acc, d_acc), p_mf1, d_mf1, wtl(p_mf1, d_mf1), p_auc, d_auc, wtl(p_auc, d_auc))
    
    print(acc, mf1, auc)

def evaluation_each_class(dataset_name):
    auc_value_all = []
    f1_value_all = []
    acc_value_all = []
    fault_type = []
    for i in range(10):
        df = pd.read_csv(f"../results/pred/{dataset_name}/{dataset_name}_pred_RF_{i}.csv")

        y_true_binary = label_binarize(df['y_test'], classes=range(6))
        y_pred_binary = label_binarize(df['BC'], classes=range(6))
        for j in range(1,6):
            auc_value = roc_auc_score(y_true_binary[:, j], y_pred_binary[:, j])
            f1_value = f1_score(y_true_binary[:, j], y_pred_binary[:, j])
            acc_value = accuracy_score(y_true_binary[:, j], y_pred_binary[:, j])
            auc_value_all.append(auc_value)
            f1_value_all.append(f1_value)
            acc_value_all.append(acc_value)
    df = pd.DataFrame()
    df['acc'] = acc_value_all
    df['f1'] = f1_value_all
    df['auc'] = auc_value_all
    df['type'] = ['NP','LE', 'DR', 'DM', 'DS']*10

    sns.set_theme(style="darkgrid")
    fig, ax =plt.subplots(1,3,constrained_layout=True)
    sns.boxplot(x='type', y='acc', data=df, ax=ax[0])
    sns.boxplot(x='type', y='f1', data=df, ax=ax[1])
    sns.boxplot(x='type', y='auc', data=df, ax=ax[2])

    for i in range(3):
        ax[i].set_xlabel("")

    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("F1-score")
    ax[2].set_ylabel("AUC")

    plt.show()
    
def motivation():
    original = pd.read_csv(f"../data2/all/fine_tune_runtime.csv")
    diff = pd.read_csv(f"../data2/all/diff_features.csv")
    
    label = pd.read_csv(f"../data2/all/label.csv")
    label['label'] = label['label'].apply(lambda x: 1 if x != 0 else x)
    sns.set_theme(style="darkgrid")
    metrics = ['loss_mean','loss_median','acc_mean','acc_median','cons_std_weight_mean', 'cons_std_weight_median','increase_loss_mean','increase_loss_median']

    fig, ax =plt.subplots(2,4,constrained_layout=True,figsize=(9, 5))
    for i in range(2):
        for j in range(4):
            print(i,j)
            df = pd.DataFrame()
            metric = list(original[metrics[i*4+j]])
            metric.extend(list(diff[metrics[i*4+j]]))
            label_l = list(label['label'])
            label_l.extend(label['label'])
            x = ['DF']*len(original)
            x.extend(['DSDM']*len(diff))

            df[i] = metric
            df['label'] = label_l
            df['x'] = x

            sns.boxplot(x='x', y=i, hue='label', data=df,ax=ax[i,j], showfliers=False,width=0.5)
            ax[i,j].set_title(metrics[i*4+j])
            ax[i,j].set_xlabel("")
            ax[i,j].set_ylabel("Value")
            ax[i,j].legend().set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.904,bottom=0.113,left=0.091,right=0.98,hspace=0.35,wspace=0.55)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, choices=['train','compare','each_class'], help='choose to train classsifiers or compare the results')
    parser.add_argument('-m1', type=str, choices=['BC','A', 'B','C'], help='metric 1')
    parser.add_argument('-m2', type=str, choices=['BC','A', 'B','C'], help='metric 2')
    args = parser.parse_args()
    if args.o == 'train':
        for dataset_name in datasets:
            print(dataset_name)
            train(dataset_name)
            evaluate(dataset_name)  
    elif args.o == 'compare':      
        hypothesis_testing(args.m1, args.m2)
    elif args.o == 'each_class':
        evaluation_each_class('all')
    # motivation()

