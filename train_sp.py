from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from cliffs_delta import cliffs_delta
import scipy.stats as stats

import pandas as pd
import numpy as np
import os
import time
from info_gain import info_gain
import argparse

dataset=['blob','circle','mnist','cifar10','reuters','imdb']
# dataset=['imdb']

from sklearn.model_selection import KFold,StratifiedKFold

def get_subset_from_index(data, index):
    new_data = []
    for i in index:
        new_data.append(data[i])
    return new_data

def get_features():
    features_all = 0
    for dataset_name in dataset:
        features_diff = pd.read_csv(f"Evaluation_sp/{dataset_name}/features_diff.csv")
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
        
        features = pd.read_csv(f"Evaluation_sp/{dataset_name}/features_old.csv")
        features.rename(columns={'model':'model_name'},inplace=True)
        d = []
        for i in features.columns:
            if 'val' in i or 'test_not_well' in i or 'test_turn_bad' in i:
                d.append(i)
        features = features.drop(axis=1,columns=d)
        merged_features = pd.merge(features_diff, features, on=['model_name','mutant','label','iter'])

        features_change = pd.read_csv(f"Evaluation_sp/{dataset_name}/change_features.csv")
        features_change = features_change.drop(axis=1,columns=['dataset_name'])          
        d = []
        for i in features_change.columns:
            if 'val' in i or 'test_not_well' in i or 'test_turn_bad' in i:
                d.append(i)
        features_change = features_change.drop(axis=1,columns=d)
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

def DT(x_train, y_train, x_test, y_test, score):
    # print('DT')
    clf = DecisionTreeClassifier(criterion='entropy', random_state=10)
    clf = clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    # print(pred, y_test)
    score['DT'] = np.append(score['DT'], accuracy_score(y_test, pred))
    return pred

def RF(x_train, y_train, x_test, y_test, score):   
    # print('RF')
    rfc = RandomForestClassifier(n_estimators = 25,criterion='entropy',random_state=0)
    rfc.fit(x_train,y_train)
    pred = rfc.predict(x_test)
    score['RF'] = np.append(score['RF'], accuracy_score(y_test, pred))
    return pred

def NB(x_train, y_train, x_test, y_test, score):
    clf = GaussianNB()
    clf = clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    score['NB'] = np.append(score['NB'], accuracy_score(y_test, pred))
    return pred

def KNN(x_train, y_train, x_test, y_test, score):
    clf = KNeighborsClassifier()
    clf = clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    score['KNN'] = np.append(score['KNN'], accuracy_score(y_test, pred))
    return pred 
  
def LR(x_train, y_train, x_test, y_test, score):
    # print('LR')
    clf = LogisticRegression(random_state=0,max_iter=10000,solver='liblinear')
    clf = clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    score['LR'] = np.append(score['LR'], accuracy_score(y_test, pred))
    return pred

def training2(features, label, train_index, test_index, score):
    features = np.array(features)
    label = np.array(label)
     
    x_train = get_subset_from_index(features, train_index)
    y_train = get_subset_from_index(label, train_index)
    x_test = get_subset_from_index(features, test_index)
    y_test = get_subset_from_index(label, test_index)

    dt_pred = DT(x_train, y_train, x_test, y_test, score)
    rf_pred = RF(x_train, y_train, x_test, y_test, score)
    nb_pred = NB(x_train, y_train, x_test, y_test, score) 
    knn_pred = KNN(x_train, y_train, x_test, y_test, score)
    lr_pred = LR(x_train, y_train, x_test, y_test, score)

    df_dict = {'dt_pred': dt_pred, 'rf_pred': rf_pred,'nb_pred': nb_pred, 'knn_pred': knn_pred, 'lr_pred': lr_pred, 'y_test': y_test}
    df = pd.DataFrame(df_dict)

    return df

def training():
    if not os.path.exists(f'Evaluation_sp/results'):
        os.makedirs(f'Evaluation_sp/results')  
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
        if not os.path.exists(f'Evaluation_sp/results/{dataset_name}'):
            os.makedirs(f'Evaluation_sp/results/{dataset_name}') 
        for i in ['A','B','C','AB','AC','BC','ABC']:
            if not os.path.exists(f'Evaluation_sp/results/{dataset_name}/{i}'):
                os.makedirs(f'Evaluation_sp/results/{dataset_name}/{i}')        
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
        
        for rs in range(0,10):
            print(rs)
            skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=rs)
            num=0
            for train_index, test_index in skf.split(new_df, new_label):
                new_train_index = []
                new_test_index = []
                for i in train_index:
                    for j in range(10):
                        new_train_index.append(i*10+j)
                for i in test_index:
                    for j in range(10):
                        new_test_index.append(i*10+j)                
                print(new_test_index)

                df = training2(features_A, label, new_train_index, new_test_index, score_A)
                df.to_csv(f'Evaluation_sp/results/{dataset_name}/A/{rs}_{num}.csv', index=False)
                df=training2(features_B, label, new_train_index, new_test_index, score_B)
                df.to_csv(f'Evaluation_sp/results/{dataset_name}/B/{rs}_{num}.csv', index=False)
                df=training2(features_C, label, new_train_index, new_test_index, score_C)
                df.to_csv(f'Evaluation_sp/results/{dataset_name}/C/{rs}_{num}.csv', index=False)
                df=training2(features_AB, label, new_train_index, new_test_index, score_AB)
                df.to_csv(f'Evaluation_sp/results/{dataset_name}/AB/{rs}_{num}.csv', index=False)
                df=training2(features_AC, label, new_train_index, new_test_index, score_AC)
                df.to_csv(f'Evaluation_sp/results/{dataset_name}/AC/{rs}_{num}.csv', index=False)
                df=training2(features_BC, label, new_train_index, new_test_index, score_BC)
                df.to_csv(f'Evaluation_sp/results/{dataset_name}/BC/{rs}_{num}.csv', index=False)
                df=training2(features_ABC, label, new_train_index, new_test_index, score_ABC)
                df.to_csv(f'Evaluation_sp/results/{dataset_name}/ABC/{rs}_{num}.csv', index=False)
                num+=1

        score_A_all[dataset_name] = score_A
        score_B_all[dataset_name] = score_B
        score_C_all[dataset_name] = score_C
        score_AB_all[dataset_name] = score_AB
        score_AC_all[dataset_name] = score_AC
        score_BC_all[dataset_name] = score_BC
        score_ABC_all[dataset_name] = score_ABC

    if not os.path.exists(f'Evaluation_sp/results'):
        os.makedirs(f'Evaluation_sp/results')    
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
                df.to_csv(f"Evaluation_sp/results/{key}_{k}.csv",index=False)


dataset_len = [1014, 936, 2028, 910, 352, 143]
# 获取每次训练和测试的数据集索引
def get_index(seed):
    features_all = get_features()
    # print(len(features_all))
    train_index_all = []
    test_index_all = []
    num = 0
    for dataset_name in dataset:
        df = features_all[features_all['dataset']==dataset_name]
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
            # print(split_num)
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

np.set_printoptions(threshold=np.inf)

def training_all():
    if not os.path.exists(f'Evaluation_sp/results/all'):
        os.makedirs(f'Evaluation_sp/results/all') 
    for i in ['A','B','C','AB','AC','BC','ABC']:
        if not os.path.exists(f'Evaluation_sp/results/all/{i}'):
            os.makedirs(f'Evaluation_sp/results/all/{i}') 
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

    # 特征排序，需要可打开
    # chi_square_(features_ABC,label)
    # reliefF_(features_ABC,label)
    # feature_selection(features_ABC,label, 'rfc')
    # rfe(features_ABC,label, 'lr')
    # cfs(features_ABC,label)
    # exit()

    # PCA_process(features_ABC,label)

    score_A = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_B = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_C = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([]),"MLP":np.array([])}
    score_AB = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_AC = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_BC = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}
    score_ABC = {"DT":np.array([]),"RF":np.array([]), "NB":np.array([]),"KNN":np.array([]),"SVM":np.array([]),"LR":np.array([])}

    for rs in range(10):
        print(f"[{rs}]")
        train_index_all, test_index_all = get_index(rs)
    # skf = StratifiedKFold(n_splits=10)
    # for train_index, test_index in skf.split(features_A, label):
        for i in range(len(train_index_all)):
            print(i)
            # print(test_index_all[i])
            # exit()
            df = training2(features_A, label, train_index_all[i], test_index_all[i], score_A)
            df.to_csv(f'Evaluation_sp/results/all/A/{rs}_{i}.csv', index=False)
            df = training2(features_B, label, train_index_all[i], test_index_all[i], score_B)
            df.to_csv(f'Evaluation_sp/results/all/B/{rs}_{i}.csv', index=False)
            df = training2(features_C, label, train_index_all[i], test_index_all[i], score_C)
            df.to_csv(f'Evaluation_sp/results/all/C/{rs}_{i}.csv', index=False)
            df = training2(features_AB, label, train_index_all[i], test_index_all[i], score_AB)
            df.to_csv(f'Evaluation_sp/results/all/AB/{rs}_{i}.csv', index=False)
            df = training2(features_AC, label, train_index_all[i], test_index_all[i], score_AC)
            df.to_csv(f'Evaluation_sp/results/all/AC/{rs}_{i}.csv', index=False)
            df = training2(features_BC, label, train_index_all[i], test_index_all[i], score_BC)
            df.to_csv(f'Evaluation_sp/results/all/BC/{rs}_{i}.csv', index=False)
            df = training2(features_ABC, label, train_index_all[i], test_index_all[i], score_ABC)
            df.to_csv(f'Evaluation_sp/results/all/ABC/{rs}_{i}.csv', index=False)
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
                df.to_csv(f"Evaluation_sp/results/{key}_{k}.csv",index=False)


# 利用汇总数据计算平均值、p值、delta
classifier = ['DT','RF','NB','KNN','LR']
m1="BC"
m2="A"
def filter_data(df, num):
    temp_list = []
    for i in range(100):
        temp_list.append(df.iloc[num+i*5])
    return pd.DataFrame(temp_list)

def ht_new(m,m1,m2):
    res_list = []
    win=0
    tie=0
    lose=0
    for i in range(5):
        for dataset_name in dataset:
            df = pd.read_csv(f"Evaluation_sp/results/{dataset_name}/{m}.csv")
            score = filter_data(df, i)
            avg_m1 = np.mean(score[m1])
            avg_m2 = np.mean(score[m2])
            # print(avg_m1)
            # print(avg_m2)
            pvalue = stats.wilcoxon(score[m1], score[m2]).pvalue
            d, res = cliffs_delta(score[m1], score[m2])
            # print(pvalue)
            # print(d)
            if pvalue < 0.05 and d >=0.147:
                result = 'win'
                win+=1
            elif pvalue < 0.05 and d <=-0.147:
                result = 'lose'
                lose+=1
            else:
                result = 'tie'
                tie+=1
            print([avg_m1, avg_m2, pvalue, d, result])
            res_list.append([avg_m1, avg_m2, pvalue, d, result])
    print(f"{win}/{tie}/{lose}")
    res_df = pd.DataFrame(res_list, columns=[m1,m2,'p','d','res'])
    res_df.to_csv(f"Evaluation_sp/results/final_{m}.csv",index=False)

def calculate_gain_ratio():
    start_time = time.time()
    f=open("gain_ratio_sp.csv","w",encoding="utf-8")
    features_all = get_features()
    # features_all.to_csv("all_data.csv",index=False)
    df = features_all
    features_A = df.iloc[:,133:261] # original
    features_B = df.iloc[:,4:132]   # diff
    features_C = df.iloc[:,261:]    # data
    label = df['label']
    features_ABC = pd.concat([features_A,features_B, features_C],join='outer',axis=1) 
    res = []
    # print(features_C.columns)
    # print(len(features_C.columns))
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
    end_time = time.time()
    print(end_time-start_time)
    f.write(f"{end_time}-{start_time}\n")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='Whether to train with each dataset separately.')
    parser.add_argument('-a', help='Whether to train with all dataset together.')
    parser.add_argument('-m1', help='metric 1')
    parser.add_argument('-m2', help='metric 2')
    args = parser.parse_args()

    if args.t:
        training()
    if args.a:
        training_all()
    if args.m1 and args.m2:
        ht_new('acc', args.m1, args.m2)
        ht_new('macro_f1', args.m1, args.m2)
        ht_new('weighted_f1', args.m1, args.m2)
