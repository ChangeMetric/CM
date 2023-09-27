from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.decomposition import PCA

from sklearn.utils import shuffle

import statsmodels.api as sm
import scipy.stats as stats
from cliffs_delta import cliffs_delta

from info_gain import info_gain
import os
import argparse

dataset=['blob','circle','mnist','cifar10','reuters','imdb']

def get_subset_from_index(data, index):
    new_data = []
    for i in index:
        new_data.append(data[i])
    return new_data

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
        d = []
        for i in features_diff.columns:
            if 'val' in i or 'test_not_well' in i or 'test_turn_bad' in i:
                d.append(i)
        features_diff = features_diff.drop(axis=1,columns=d)
        # print(features_diff)
        
        features = pd.read_csv(f"Evaluation/{dataset_name}/features_old.csv")
        features.rename(columns={'model':'model_name'},inplace=True)
        d = []
        for i in features.columns:
            if 'val' in i or 'test_not_well' in i or 'test_turn_bad' in i:
                d.append(i)
        features = features.drop(axis=1,columns=d)
        merged_features = pd.merge(features_diff, features, on=['model_name','mutant','label','iter'])  
        

        features_change = pd.read_csv(f"Evaluation/{dataset_name}/change_features.csv")
        features_change = features_change.drop(axis=1,columns=['dataset_name'])          
        d = []
        for i in features_change.columns:
            if 'val' in i or 'test_not_well' in i or 'test_turn_bad' in i:
                d.append(i)
        features_change = features_change.drop(axis=1,columns=d)
        merged_features = pd.merge(merged_features, features_change, on=['model_name','mutant','iter'])

        # merged_features = pd.concat([features_diff,features,features_change],join='outer',axis=1)



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
    # print(features_all)

def DT(x_train, y_train, x_test, y_test, score):
    # print('DT')
    clf = DecisionTreeClassifier(criterion='entropy', random_state=10)
    clf = clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    # print(pred, y_test)
    score['DT'] = np.append(score['DT'], f1_score(y_test, pred))
    score['DT'] = np.append(score['DT'], accuracy_score(y_test, pred))
    score['DT'] = np.append(score['DT'], roc_auc_score(y_test, pred))
    return clf

def RF(x_train, y_train, x_test, y_test, score):   
    # print('RF')
    rfc = RandomForestClassifier(n_estimators = 25,criterion='entropy',random_state=0)
    rfc.fit(x_train,y_train)
    pred = rfc.predict(x_test)
    score['RF'] = np.append(score['RF'], f1_score(y_test, pred))
    score['RF'] = np.append(score['RF'], accuracy_score(y_test, pred))
    score['RF'] = np.append(score['RF'], roc_auc_score(y_test, pred))
    return rfc

def NB(x_train, y_train, x_test, y_test, score):
    clf = GaussianNB()
    clf = clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    score['NB'] = np.append(score['NB'], f1_score(y_test, pred))
    score['NB'] = np.append(score['NB'], accuracy_score(y_test, pred))
    score['NB'] = np.append(score['NB'], roc_auc_score(y_test, pred))
    return clf

def KNN(x_train, y_train, x_test, y_test, score):
    clf = KNeighborsClassifier()
    clf = clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    score['KNN'] = np.append(score['KNN'], f1_score(y_test, pred))
    score['KNN'] = np.append(score['KNN'], accuracy_score(y_test, pred))
    score['KNN'] = np.append(score['KNN'], roc_auc_score(y_test, pred))
    return clf      

def LR(x_train, y_train, x_test, y_test, score):
    clf = LogisticRegression(random_state=0,max_iter=10000,solver='liblinear')
    clf = clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    score['LR'] = np.append(score['LR'], f1_score(y_test, pred))
    score['LR'] = np.append(score['LR'], accuracy_score(y_test, pred))
    score['LR'] = np.append(score['LR'], roc_auc_score(y_test, pred))
    return clf

def PCA_process(features):
    pca = PCA(n_components=0.9999)
    features_new = pca.fit_transform(features)
    return features_new, pca

np.set_printoptions(threshold=np.inf)


def training2(features, label, train_index, test_index, score):
    features = np.array(features)
    # features_new = PCA_process(features)
    # print(features.shape,np.array(features_new).shape)
    label = np.array(label)
     
    x_train = get_subset_from_index(features, train_index)
    y_train = get_subset_from_index(label, train_index)
    x_test = get_subset_from_index(features, test_index)
    y_test = get_subset_from_index(label, test_index)

    # x_train, pca = PCA_process(x_train)
    # x_test = pca.transform(x_test)

    DT(x_train, y_train, x_test, y_test, score)
    RF(x_train, y_train, x_test, y_test, score)
    NB(x_train, y_train, x_test, y_test, score) 
    KNN(x_train, y_train, x_test, y_test, score)
    LR(x_train, y_train, x_test, y_test, score)

def my_shuffle(features_A, features_B, features_AB, features_AC, features_BC, features_ABC, label, seed):
    features_A_array = np.array(features_A)
    features_B_array = np.array(features_B)
    features_AB_array = np.array(features_AB)
    features_AC_array = np.array(features_AC)
    features_BC_array = np.array(features_BC)
    features_ABC_array = np.array(features_ABC)
    label_array = np.array(label)

    np.random.seed(seed)
    np.random.shuffle(features_A_array)
    np.random.seed(seed)
    np.random.shuffle(features_B_array)
    np.random.seed(seed)
    np.random.shuffle(features_AB_array)
    np.random.seed(seed)
    np.random.shuffle(features_AC_array)
    np.random.seed(seed)
    np.random.shuffle(features_BC_array)
    np.random.seed(seed)
    np.random.shuffle(features_ABC_array)
    np.random.seed(seed)
    np.random.shuffle(label_array)
    return features_A_array,features_B_array,features_AB_array,features_AC_array,features_BC_array,features_ABC_array,label_array

def training():
    if not os.path.exists(f'Evaluation/results'):
        os.makedirs(f'Evaluation/results')  
    features_all = get_features()
    score_A_all = {}
    score_B_all = {}
    score_C_all = {}
    score_AB_all = {}
    score_AC_all = {}
    score_BC_all = {}
    score_ABC_all = {}
    for dataset_name in dataset:
        print(dataset_name)
        df = features_all[features_all['dataset']==dataset_name]

        features_A = df.iloc[:,133:261] # original
        features_B = df.iloc[:,4:132]   # diff
        features_C = df.iloc[:,261:]    # data

        label = df['label']
        
        features_AB = pd.concat([features_A,features_B],join='outer',axis=1)
        features_AC = pd.concat([features_A,features_C],join='outer',axis=1)
        features_BC = pd.concat([features_B,features_C],join='outer',axis=1)
        features_ABC = pd.concat([features_A, features_B,features_C],join='outer',axis=1)

        # 每10次训练取1次
        new_df_list = []
        for i in range(int(len(df)/10)):
            new_df_list.append(df.loc[i*10])
        new_df = pd.DataFrame(new_df_list)
        new_features_A = new_df.iloc[:,133:261] # original
        new_features_B = new_df.iloc[:,4:132]   # diff
        new_features_C = new_df.iloc[:,261:]    # data
        new_label = new_df['label']

        new_features_AB = pd.concat([new_features_A,new_features_B],join='outer',axis=1)
        new_features_AC = pd.concat([new_features_A,new_features_C],join='outer',axis=1)
        new_features_BC = pd.concat([new_features_B,new_features_C],join='outer',axis=1)
        new_features_ABC = pd.concat([new_features_A, new_features_B,new_features_C],join='outer',axis=1)

        score_A = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
        score_B = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
        score_C = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
        score_AB = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
        score_AC = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
        score_BC = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
        score_ABC = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
        
        for i in range(0,10):
            print(i)
            skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
            for train_index, test_index in skf.split(new_df, new_label):
                new_train_index = []
                new_test_index = []
                for i in train_index:
                    for j in range(10):
                        new_train_index.append(i*10+j)
                for i in test_index:
                    for j in range(10):
                        new_test_index.append(i*10+j)                
                # print(test_index)
                print(new_test_index)

                training2(features_A, label, new_train_index, new_test_index, score_A)
                training2(features_B, label, new_train_index, new_test_index, score_B)
                training2(features_C, label, new_train_index, new_test_index, score_C)
                training2(features_AB, label, new_train_index, new_test_index, score_AB)
                training2(features_AC, label, new_train_index, new_test_index, score_AC)
                training2(features_BC, label, new_train_index, new_test_index, score_BC)
                training2(features_ABC, label, new_train_index, new_test_index, score_ABC)
            
        # exit()


        score_A_all[dataset_name] = score_A
        score_B_all[dataset_name] = score_B
        score_C_all[dataset_name] = score_C
        score_AB_all[dataset_name] = score_AB
        score_AC_all[dataset_name] = score_AC
        score_BC_all[dataset_name] = score_BC
        score_ABC_all[dataset_name] = score_ABC

    for key in list(score_A_all.keys()):
        for k, v in score_A_all[key].items():
            if k in ['DT','RF','NB','KNN','SVM','LR']:
                res = {}
                res['A'] = score_A_all[key][k]
                res['B'] =score_B_all[key][k]
                res['C'] =score_C_all[key][k]
                res['AB'] =score_AB_all[key][k]
                res['AC'] =score_AC_all[key][k]
                res['BC'] =score_BC_all[key][k]
                res['ABC'] =score_ABC_all[key][k]
                df = pd.DataFrame(res)
                df.to_csv(f"Evaluation/results/{key}_{k}.csv",index=False)




dataset_len = [234, 216, 468, 210, 192, 78]
# 获取每次训练和测试的数据集索引
def get_index(seed):
    features_all = get_features()
    train_index_all = []
    test_index_all = []
    num = 0
    for dataset_name in dataset:
        df = features_all[features_all['dataset']==dataset_name]
        # print(len(df))
        label = df['label']
        features_A = df.iloc[:,133:261] # original

        # 每10次训练取1次
        new_df_list = []
        for i in range(int(len(df)/10)):
            new_df_list.append(df.loc[i*10])
        new_df = pd.DataFrame(new_df_list)
        new_features_A = new_df.iloc[:,133:261] # original
        new_label = new_df['label']

        skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
        split_num = 0
        # for train_index, test_index in skf.split(features_A, label):
        for train_index, test_index in skf.split(new_features_A, new_label):
            if num == 0:
                train_index_all.append(train_index)
                test_index_all.append(test_index)
            else:
                offset = 0
                for i in range(num):
                    offset+=dataset_len[i]
                new_train_index = train_index+offset
                new_test_index = test_index+offset
                train_index_all[split_num] = np.concatenate((train_index_all[split_num], new_train_index),axis=0)
                test_index_all[split_num] = np.concatenate((test_index_all[split_num], new_test_index),axis=0)
            split_num+=1
        num+=1

    new_train_index = []
    new_test_index = []
    for iter in range(10):
        temp_train = []
        temp_test = []
        for i in train_index_all[iter]:
            for j in range(10):
                temp_train.append(i*10+j)
        for i in test_index_all[iter]:
            for j in range(10):
                temp_test.append(i*10+j)  
        new_train_index.append(temp_train)
        new_test_index.append(temp_test)            
    return new_train_index, new_test_index

def training_all():
    features_all = get_features()
    score_A_all = {}
    score_B_all = {}
    score_C_all = {}
    score_AB_all = {}
    score_AC_all = {}
    score_BC_all = {}
    score_ABC_all = {}

    df = features_all
    features_A = df.iloc[:,133:261] # original
    features_B = df.iloc[:,4:132]   # diff
    features_C = df.iloc[:,261:]    # data
    label = df['label']

    features_AB = pd.concat([features_A,features_B],join='outer',axis=1)
    features_AC = pd.concat([features_A,features_C],join='outer',axis=1)
    features_BC = pd.concat([features_B,features_C],join='outer',axis=1)
    features_ABC = pd.concat([features_A, features_B,features_C],join='outer',axis=1)

    score_A = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_B = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_C = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_AB = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_AC = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_BC = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_ABC = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}

    for i in range(10):
        print(i)
        train_index_all, test_index_all = get_index(i)
    # skf = StratifiedKFold(n_splits=10)
    # for train_index, test_index in skf.split(features_A, label):
        for i in range(len(train_index_all)):
            # print(test_index_all[i])
            # exit()
            training2(features_A, label, train_index_all[i], test_index_all[i], score_A)
            training2(features_B, label, train_index_all[i], test_index_all[i], score_B)
            training2(features_C, label, train_index_all[i], test_index_all[i], score_C)
            training2(features_AB, label, train_index_all[i], test_index_all[i], score_AB)
            training2(features_AC, label, train_index_all[i], test_index_all[i], score_AC)
            training2(features_BC, label, train_index_all[i], test_index_all[i], score_BC)
            training2(features_ABC, label, train_index_all[i], test_index_all[i], score_ABC)
        # print(test_index_all[2][5])
    # exit()

    score_A_all['all'] = score_A
    score_B_all['all'] = score_B
    score_C_all['all'] = score_C
    score_AB_all['all'] = score_AB
    score_AC_all['all'] = score_AC
    score_BC_all['all'] = score_BC
    score_ABC_all['all'] = score_ABC

    for key in list(score_A_all.keys()):
        for k, v in score_A_all[key].items():
            if k in ['DT','RF','NB','KNN','LR']:
                res = {}
                res['A'] = score_A_all[key][k]
                res['B'] =score_B_all[key][k]
                res['C'] =score_C_all[key][k]
                res['AB'] =score_AB_all[key][k]
                res['AC'] =score_AC_all[key][k]
                res['BC'] =score_BC_all[key][k]
                res['ABC'] =score_ABC_all[key][k]
                df = pd.DataFrame(res)
                df.to_csv(f"Evaluation/results/{key}_{k}.csv",index=False)

# ---- 分析 ----
classifier = ['DT','RF','NB','KNN','LR']

# 算F1均值
def calculate_f1():
    mean_data_f1 = []; mean_data_acc = []; mean_data_roc = []
    for classifier_name in classifier:
        for dataset_name in dataset:
            df = pd.read_csv(f"Evaluation/results/{dataset_name}_{classifier_name}.csv")
            f1 = []; acc = []; roc = []
            for i in range(100):
                f1.append(df.iloc[i*3])
                acc.append(df.iloc[1+i*3])
                roc.append(df.iloc[2+i*3])
            df_f1 = pd.DataFrame(f1)
            df_acc = pd.DataFrame(acc)
            df_roc = pd.DataFrame(roc)
            mean_data_f1.append(df_f1.mean())
            mean_data_acc.append(df_acc.mean())
            mean_data_roc.append(df_roc.mean())
        df = pd.read_csv(f"Evaluation/results/all_{classifier_name}.csv")
        f1 = []; acc = []; roc = []
        for i in range(100):
            f1.append(df.iloc[i*3])
            acc.append(df.iloc[1+i*3])
            roc.append(df.iloc[2+i*3])
        df_f1 = pd.DataFrame(f1)
        df_acc = pd.DataFrame(acc)
        df_roc = pd.DataFrame(roc)
        mean_data_f1.append(df_f1.mean())
        mean_data_acc.append(df_acc.mean())
        mean_data_roc.append(df_roc.mean())
    mean_data_df_f1 = pd.DataFrame(mean_data_f1)
    mean_data_df_acc = pd.DataFrame(mean_data_acc)
    mean_data_df_roc = pd.DataFrame(mean_data_roc)
    mean_data_df_f1.to_csv("Evaluation/results/f1.csv",index=False)
    mean_data_df_acc.to_csv("Evaluation/results/acc.csv",index=False)
    mean_data_df_roc.to_csv("Evaluation/results/roc.csv",index=False)

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

# 假设检验，BC vs A
m1 = 'BC'
m2 = 'A'
def hypothesis_testing(metric, m1, m2):
    print(metric)
    f = open(f"Evaluation/results/HT_{metric}.csv","w",encoding="utf-8")
    all_win=0
    all_tie=0
    all_lose=0
    for classifier_name in classifier:
        win=0
        tie=0
        lose=0
        for dataset_name in dataset:
            df = pd.read_csv(f"Evaluation/results/{dataset_name}_{classifier_name}.csv")
            df = filter_data(df, metric)
            # print(df)
            pvalue = stats.wilcoxon(df[m1], df[m2]).pvalue
            d, res = cliffs_delta(df[m1], df[m2])
            if pvalue < 0.05 and d >=0.147:
                res = 'win'
                win+=1
            elif pvalue < 0.05 and d <=-0.147:
                res = 'lose'
                lose+=1
            else:
                res = 'tie'
                tie+=1
            f.write(f"{classifier_name},{dataset_name},{pvalue},{d},{res}\n")
            print(classifier_name, dataset_name, pvalue, d,res)
            
        df = pd.read_csv(f"Evaluation/results/all_{classifier_name}.csv")
        df = filter_data(df, metric)
        pvalue = stats.wilcoxon(df[m1], df[m2]).pvalue
        d, res = cliffs_delta(df[m1], df[m2])
        # print(type(d))
        if pvalue < 0.05 and d >=0.147:
            res = 'win'
            win+=1
        elif pvalue < 0.05 and d <=-0.147:
            res = 'lose'
            lose+=1
        else:
            res = 'tie'
            tie+=1
        f.write(f"{classifier_name},'all',{pvalue},{d},{res}\n")
        print(classifier_name, 'all', pvalue, d, res)
        print(f"{win}/{tie}/{lose}")
        all_win+=win
        all_tie+=tie
        all_lose+=lose
    print(f"{all_win}/{all_tie}/{all_lose}")
    return all_win,all_tie,all_lose


def calculate_gain_ratio():
    f=open("gain_ratio.csv","w",encoding="utf-8")
    features_all = get_features()
    # features_all.to_csv("all_data.csv",index=False)
    df = features_all
    features_A = df.iloc[:,133:261] # original
    features_B = df.iloc[:,4:132]   # diff
    features_C = df.iloc[:,261:]    # data
    print(features_A.columns)
    print(features_B.columns)
    print(features_C.columns)
    label = df['label']
    features_ABC = pd.concat([features_A,features_B],join='outer',axis=1) 
    res = []
    num=1
    for columns in features_ABC.columns:
        num+=1
        gr = info_gain.info_gain_ratio(label, features_ABC[columns])
        print(num, columns, gr)
        res.append([gr, columns])
    res.sort(key=lambda x:x[0])
    for i in res:
        f.write(f"{i[1]},{i[0]}\n")
        print(f"{i[0]}\t{i[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='Whether to train with each dataset separately.')
    parser.add_argument('-a', help='Whether to train with all dataset together.')
    parser.add_argument('-e', help='Whether to summarize the results.')
    parser.add_argument('-m1', help='metric 1')
    parser.add_argument('-m2', help='metric 2')
    args = parser.parse_args()

    if args.t:
        training()
    if args.a:
        training_all()
    if args.e:
        calculate_f1()
    if args.m1 and args.m2:
        win1, tie1, lose1 = hypothesis_testing('f1', args.m1, args.m2)
        win2, tie2, lose2 = hypothesis_testing('acc', args.m1, args.m2)
        win3, tie3, lose3 = hypothesis_testing('roc', args.m1, args.m2)
        print(f"{win1+win2+win3}, {tie1+tie2+tie3}, {lose1+lose2+lose3}")
