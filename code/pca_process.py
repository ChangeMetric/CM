from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd
import numpy as np
import os
import time
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

classifiers = ['DT','RF','NB','KNN','LR']
class ClassifierPCA:
    def __init__(self, features, label) -> None:
        self.label = label
        self.features = features

        self.classifiers = {}
        self.pred = {}
        self.classification_reports = {}
    
    def my_split(self, seed):
        my_pca = PCA(n_components=0.9999)
        self.features = my_pca.fit_transform(self.features)
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
    features_BC = pd.concat([features_B,features_C],join='outer',axis=1) 

    s_time=time.time()
    for i in range(10):
        clf_bc = ClassifierPCA(features_BC,label)
        clf_bc.my_split(i)
        for clf_name in classifiers:
            clf_bc.train_pred(clf_name)

            df = pd.DataFrame()
            df['BC'] = clf_bc.pred[clf_name]
            df['y_test'] = list(clf_bc.y_test)

            if not os.path.exists(f"../results/pred/pca/{dataset_name}"):
                os.makedirs(f"../results/pred/pca/{dataset_name}")
            df.to_csv(f"../results/pred/pca/{dataset_name}/{dataset_name}_pred_{clf_name}_{i}.csv",index=False)

    e_time=time.time()
    with open(f"../results/pred/pca/{dataset_name}/time.txt","w",encoding="utf-8") as f:
        f.write(f"{e_time-s_time}")

metrics = ['BC']
def evaluate(dataset_name):
    res_acc = []
    res_mf1 = []
    res_auc = []
    for clf_name in classifiers:
        # print(clf_name)
        for i in range(10):
            # print(i)
            df = pd.read_csv(f"../results/pred/pca/{dataset_name}/{dataset_name}_pred_{clf_name}_{i}.csv")
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
    df_acc = pd.DataFrame(res_acc, columns=['index','BC'])
    df_acc.to_csv(f"../results/evaluation/pca/{dataset_name}_acc.csv",index=False)
    df_mf1 = pd.DataFrame(res_mf1, columns=['index','BC'])
    df_mf1.to_csv(f"../results/evaluation/pca/{dataset_name}_mf1.csv",index=False)
    df_auc = pd.DataFrame(res_auc, columns=['index','BC'])
    df_auc.to_csv(f"../results/evaluation/pca/{dataset_name}_auc.csv",index=False)

datasets=['blob','circle','mnist','cifar10','reuters','imdb','all']
def compare():
    sns.set_theme(style="darkgrid")
    df = pd.DataFrame() 
    acc = []
    mf1 = []
    auc = []
    pca_or_not = []
    dataset = []
    for dataset_name in datasets:
        df_acc_pca = pd.read_csv(f"../results/evaluation/pca/{dataset_name}_mf1.csv")
        df_mf1_pca = pd.read_csv(f"../results/evaluation/pca/{dataset_name}_mf1.csv")
        df_auc_pca = pd.read_csv(f"../results/evaluation/pca/{dataset_name}_auc.csv")
        df_acc = pd.read_csv(f"../results/evaluation/{dataset_name}_acc.csv")
        df_mf1 = pd.read_csv(f"../results/evaluation/{dataset_name}_mf1.csv")
        df_auc = pd.read_csv(f"../results/evaluation/{dataset_name}_auc.csv")

        acc.extend(list(df_acc_pca['BC'].iloc[10:20]))
        acc.extend(list(df_acc['BC'].iloc[10:20]))
        mf1.extend(list(df_mf1_pca['BC'].iloc[10:20]))
        mf1.extend(list(df_mf1['BC'].iloc[10:20]))
        auc.extend(list(df_auc_pca['BC'].iloc[10:20]))
        auc.extend(list(df_auc['BC'].iloc[10:20]))
        pca_or_not.extend(['with PCA']*10)
        pca_or_not.extend(['without PCA']*10)
        dataset.extend([dataset_name]*20)
    df['acc'] = acc
    df['mf1'] = mf1
    df['auc'] = auc
    df['pca_or_not'] = pca_or_not
    df['dataset'] = dataset

    fig, ax =plt.subplots(1,3,constrained_layout=True)
    sns.boxplot(x='dataset', y='acc', hue='pca_or_not', data=df, ax=ax[0])
    sns.boxplot(x='dataset', y='mf1', hue='pca_or_not', data=df, ax=ax[1])
    sns.boxplot(x='dataset', y='auc', hue='pca_or_not', data=df, ax=ax[2])
    for i in range(3):
        ax[i].legend().set_visible(False)
        new_ticks = [0, 1, 2, 3, 4, 5, 6]
        new_labels = ['Blob','Circle','MNIST','CIFAR-10','Reuters','IMDb','All']
        ax[i].set_xticks(new_ticks)
        ax[i].set_xticklabels(new_labels,rotation=90)
        ax[i].set_xlabel("")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Macro-F1")
    ax[2].set_ylabel("AUC")
    # plt.legend(False)
    plt.show()

if __name__ == "__main__":
    # for dataset_name in datasets:
    #     train(dataset_name)
    #     evaluate(dataset_name)
    compare()