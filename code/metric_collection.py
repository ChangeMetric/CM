import os
import time 
import pandas as pd
import numpy as np
from Utils.isKilled import is_diff_sts 
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
from scipy.spatial import distance

datasets = ['blob']

model_num = {'blob':39,'circle':36, 'cifar10':35, 'mnist':78, 'reuters':32,'imdb':13}

def extract_feature(df: pd.DataFrame):
    feature_dict = {}
    OPERATORS = ["mean", "std", "skew", "median", "var", "sem", "max", "min"]
    features = {k: OPERATORS for k in df.columns}
    extracted_feat = df.agg(features).to_dict()

    for para, values in extracted_feat.items():
        for p, v in values.items():
            key = "{}_{}".format(para, p)

            # handle exceptional value
            if type(v) == str and (v == "0" or v == "False" or v == "FALSE"):
                v = 0.0

            if type(v) == str and (v == "1" or v == "True" or v == "TRUE"):
                v = 1.0

            if type(v) == np.bool_ and v == np.bool_(False):
                v = 0.0
            if type(v) == np.bool_ and v == np.bool_(True):
                v = 1.0

            # if type(v) != float:
            #     print("Type", type(v), v, key)
            feature_dict[key] = v
    return feature_dict

# 计算20×8个度量
def summary_pre_train(dataset_name):
    dataset_summary_dict = {}
    for num in range(model_num[dataset_name]):
        # dataset_summary_dict = {}
        print(f"summary {dataset_name} {num}")
        df = pd.read_csv(f"../models/{dataset_name}/{num}/monitor_features/pre_train/monitor_features.csv")
        df = df.fillna(0.0)
        df = df.replace('False', 0)
        df = df.astype('float')
        feature_dict = extract_feature(df)
        for key, value in feature_dict.items():
            if key not in dataset_summary_dict:
                dataset_summary_dict[key] = [value]
            else:
                dataset_summary_dict[key].append(value)
    new_df = pd.DataFrame(dataset_summary_dict)
    new_df.to_csv(f"../data2/{dataset_name}/pre_train_runtime.csv",index=False)

def summary_fine_tune(dataset_name):
    dataset_summary_dict = {}
    for num in range(model_num[dataset_name]):
        print(f"summary {dataset_name} {num}")
        for ft_num in range(6):
            for i in range(5):
                df = pd.read_csv(f"../models/{dataset_name}/{num}/monitor_features/fine_tune_{ft_num}/{i}/monitor_features.csv")
                df = df.fillna(0.0)
                df = df.replace('False', 0)
                df = df.astype('float')
                feature_dict = extract_feature(df)
                for key, value in feature_dict.items():
                    if key not in dataset_summary_dict:
                        dataset_summary_dict[key] = [value]
                    else:
                        dataset_summary_dict[key].append(value)
    new_df = pd.DataFrame(dataset_summary_dict)
    new_df.to_csv(f"../data2/{dataset_name}/fine_tune_runtime.csv",index=False)    

# 计算标签
def calculate_is_diff(dataset_name):
    results_list = []
    for num in range(model_num[dataset_name]):
        print(dataset_name,num)
        orig_accuracy_list = []
        for i in range(5):
            with open(f"../models/{dataset_name}/{num}/monitor_features/fine_tune_0/{i}/test_results.txt", "r", encoding="utf-8") as f:
                orig_accuracy_list.append(float(f.read().replace("\n","").split(",")[1]))
        avg_origin_acc = sum(orig_accuracy_list) / len(orig_accuracy_list)    

        for i in range(5):
            results_list_tmp = [dataset_name, num, f"ft_0", i, 0]   
            results_list.append(results_list_tmp)                 

        for ft_num in [1,2,3,4,5]:
            accuracy_list = []
            for i in range(5):
                with open(f"../models/{dataset_name}/{num}/monitor_features/fine_tune_{ft_num}/{i}/test_results.txt", "r", encoding="utf-8") as f:
                    accuracy_list.append(float(f.read().replace("\n","").split(",")[1]))
            is_kill = is_diff_sts(orig_accuracy_list, accuracy_list)    # 是否有显著性差异 
            avg_cur_acc = sum(accuracy_list) / len(accuracy_list)
            is_faulty = int(is_kill and avg_cur_acc < avg_origin_acc)

            if is_faulty:
                for i in range(5):
                    # results_list_tmp = [model_dir, f"ft_{ft_num}", i, ft_num//10]
                    results_list_tmp = [dataset_name, num, f"ft_{ft_num}", i, ft_num]  
                    results_list.append(results_list_tmp)
            else:
                for i in range(5):
                    results_list_tmp = [dataset_name, num, f"ft_{ft_num}", i, 0]   
                    results_list.append(results_list_tmp)                    
    test = pd.DataFrame(columns = ['dataset_name','model_num','mutant_type','iter','label'],data=results_list)
    test.to_csv(f'../data2/{dataset_name}/label.csv',index=False)

def get_diff_features(dataset_name):
    pre_train = pd.read_csv(f"../data2/{dataset_name}/pre_train_runtime.csv")
    fine_tune = pd.read_csv(f"../data2/{dataset_name}/fine_tune_runtime.csv")
    diff_features = pd.DataFrame(columns=pre_train.columns)
    for i in range(len(pre_train)):
        pre_train_item = pre_train.iloc[i]
        for j in range(30):
            fine_tune_item = fine_tune.iloc[i*30+j]
            diff_features.loc[i*30+j] = pre_train_item-fine_tune_item

    diff_features.to_csv(f"../data2/{dataset_name}/diff_features.csv",index=False)


def mmd_rbf(X, Y, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def preprocess(df):
    df = df.fillna(0.0)
    df = df.replace('False', 0)
    df = df.astype('float')
    return df

def mdcm(dataset_name):
    res = []
    for num in range(model_num[dataset_name]):
        pre_train_data = pd.read_csv(f"../models/{dataset_name}/{num}/monitor_features/pre_train/monitor_features.csv")
        pre_train_data = preprocess(pre_train_data)
        for ft_num in range(6):
            for i in range(5):
                ft_data = pd.read_csv(f"../models/{dataset_name}/{num}/monitor_features/fine_tune_{ft_num}/{i}/monitor_features.csv")
                ft_data = preprocess(ft_data)
                res_item = []
                for j in pre_train_data.columns:
                    statistic, p_value = stats.ks_2samp(pre_train_data[j], ft_data[j])

                    if len(pre_train_data[j]) < 5:
                        pre_train_data_ = pre_train_data[j]
                        ft_data_ = ft_data[j][:len(pre_train_data[j])]
                    else:
                        pre_train_data_ = pre_train_data[j][-5:]
                        ft_data_ = ft_data[j]

                    cs = cosine_similarity(np.array(pre_train_data_).reshape(1,-1), np.array(ft_data_).reshape(1,-1))[0][0]
                    euclidean_distance = distance.euclidean(pre_train_data_, ft_data_)
                    manhatttan_distance = np.sum(np.abs(np.array(pre_train_data_)-np.array(ft_data_)))
                    mmd = mmd_rbf(np.array(pre_train_data_).reshape(1,-1), np.array(ft_data_).reshape(1,-1))

                    res_item.extend([p_value, cs, euclidean_distance, manhatttan_distance, mmd]) 
                res.append(res_item)

    columns_name = []
    change_metrics = ["ks_p", "cosine_similarity", "euclidean_distance", "manhatttan_distance", "mmd_rbf"]
    for i in pre_train_data.columns:
        for metrics in change_metrics:
            columns_name.append(f"{i}_{metrics}")
    
    df=pd.DataFrame(res, columns=columns_name)
    df.to_csv(f"../data2/{dataset_name}/diff_features2.csv",index=False)

def combine_all():
    datasets=['blob','circle','mnist','cifar10','reuters','imdb']
    features_As = []
    features_Bs = []
    features_Cs = []
    labels = []
    for dataset_name in datasets:
        features_A = pd.read_csv(f"../data2/{dataset_name}/fine_tune_runtime.csv")
        features_B = pd.read_csv(f"../data2/{dataset_name}/diff_features.csv")
        features_C = pd.read_csv(f"../data2/{dataset_name}/diff_features2.csv")
        label = pd.read_csv(f"../data2/{dataset_name}/label.csv")
        features_As.append(features_A)
        features_Bs.append(features_B)
        features_Cs.append(features_C)
        labels.append(label)
    features_A_all = pd.concat(features_As,join='outer',axis=0) 
    features_B_all = pd.concat(features_Bs,join='outer',axis=0) 
    features_C_all = pd.concat(features_Cs,join='outer',axis=0) 
    label_all = pd.concat(labels,join='outer',axis=0) 

    features_A_all.to_csv(f"../data2/all/fine_tune_runtime.csv",index=False)
    features_B_all.to_csv(f"../data2/all/diff_features.csv",index=False)
    features_C_all.to_csv(f"../data2/all/diff_features2.csv",index=False)
    label_all.to_csv(f"../data2/all/label.csv",index=False)

if __name__ == "__main__":
    # for dataset_name in ['blob','circle','mnist','cifar10','reuters','imdb']:
    # print(dataset_name)
    # dataset_name = 'circle'
    # s_time = time.time()
    # summary_pre_train(dataset_name)
    # summary_fine_tune(dataset_name)
    # calculate_is_diff(dataset_name)
    # get_diff_features(dataset_name)
    # mdcm(dataset_name)
    # e_time = time.time()
    # with open(f"../data2/{dataset_name}/time.txt","w",encoding="utf-8") as f:
    #     f.write(f"{e_time-s_time}")

    combine_all()