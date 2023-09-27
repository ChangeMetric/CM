import pandas as pd
import numpy as np
import os

from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

import scipy.stats as stats
from scipy.spatial import distance
from scipy.stats import skew, sem
import argparse

datasets=['blob','circle','mnist','cifar10','reuters','imdb']
# datasets=['imdb']

# refer to https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def preprocess(df):
    df = df.fillna(0.0)
    df = df.replace('False', 0)
    df = df.astype('float')
    df.drop(['not_converge', 'unstable_loss'], axis=1, inplace=True)
    return df


change_metrics = ["ks_p", "cosine_similarity", "euclidean_distance", "manhatttan_distance", "mmd_rbf"]
operators_func = [np.mean, np.std, skew, np.median, np.var, sem, np.max, np.min]

def main(base):
    for dataset_name in datasets:
        results = []
        model_num = 1
        for model_name in os.listdir(f"Evaluation_sp/{dataset_name}"):
            if os.path.isfile(f"Evaluation_sp/{dataset_name}/{model_name}"):
                continue
            print(model_num, dataset_name, model_name)
            model_num+=1

            first_train_data = pd.read_csv(f"Evaluation/{dataset_name}/{model_name}/results/first_train/monitor_features.csv")
            first_train_data = preprocess(first_train_data)
            if base == "Evaluation":
                # print("ft")
                for iter_num in range(10):
                    # print(iter_num)
                    results_item = [dataset_name,model_name,'ft', iter_num]
                    ft_data = pd.read_csv(f"Evaluation/{dataset_name}/{model_name}/results/ft/{iter_num}/monitor_features.csv")
                    ft_data = preprocess(ft_data)
                    for i in first_train_data.columns:

                        # 对向量维度无要求的度量
                        statistic, p_value = stats.ks_2samp(first_train_data[i], ft_data[i])

                        # 要求向量维度一致的度量
                        if len(first_train_data[i]) < 5:
                            first_train_data_ = first_train_data[i]
                            ft_data_ = ft_data[i][:len(first_train_data[i])]
                        else:
                            first_train_data_ = first_train_data[i][-5:]
                            ft_data_ = ft_data[i]

                        cs = cosine_similarity(np.array(first_train_data_).reshape(1,-1), np.array(ft_data_).reshape(1,-1))[0][0]
                        euclidean_distance = distance.euclidean(first_train_data_, ft_data_)
                        manhatttan_distance = np.sum(np.abs(np.array(first_train_data_)-np.array(ft_data_)))
                        mmd = mmd_rbf(np.array(first_train_data_).reshape(1,-1), np.array(ft_data_).reshape(1,-1))

                        results_item.extend([p_value, cs, euclidean_distance, manhatttan_distance, mmd])
                    results.append(results_item)

            if base == 'Evaluation':
                ft_num_all= [0,1,2,3,4]
            else:
                if dataset_name in ['reuters','imdb']:
                    ft_num_all= [10,11,20,21,30,31,40,41,50,51]
                else:
                    ft_num_all= [10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44,50,51,52,53,54]
            for ft_num in ft_num_all:
                # print(f"ft_num{ft_num}")
                for iter_num in range(10):
                    # print(iter_num)
                    results_item = [dataset_name,model_name,f'ft_{ft_num}', iter_num]
                    ft_data = pd.read_csv(f"{base}/{dataset_name}/{model_name}/results/ft_{ft_num}/{iter_num}/monitor_features.csv")
                    ft_data = preprocess(ft_data)
                    for i in first_train_data.columns:
                        statistic, p_value = stats.ks_2samp(first_train_data[i], ft_data[i])

                        if len(first_train_data[i]) < 5:
                            first_train_data_ = first_train_data[i]
                            ft_data_ = ft_data[i][:len(first_train_data[i])]
                        else:
                            first_train_data_ = first_train_data[i][-5:]
                            ft_data_ = ft_data[i]

                        cs = cosine_similarity(np.array(first_train_data_).reshape(1,-1), np.array(ft_data_).reshape(1,-1))[0][0]
                        euclidean_distance = distance.euclidean(first_train_data_, ft_data_)
                        manhatttan_distance = np.sum(np.abs(np.array(first_train_data_)-np.array(ft_data_)))
                        mmd = mmd_rbf(np.array(first_train_data_).reshape(1,-1), np.array(ft_data_).reshape(1,-1))

                        results_item.extend([p_value, cs, euclidean_distance, manhatttan_distance, mmd])
                    results.append(results_item)                
        
        columns_name = ['dataset_name','model_name','mutant','iter']
        for i in first_train_data.columns:
            for metrics in change_metrics:
                columns_name.append(f"{i}_{metrics}")

        results_df = pd.DataFrame(results, columns = columns_name)
        results_df.to_csv(f"{base}/{dataset_name}/change_features.csv", index= False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', '-bs', default='Evaluation', choices=['Evaluation','Evaluation_sp'], help='Base directory. Default: Evaluation')
    args = parser.parse_args()

    main(args.base)