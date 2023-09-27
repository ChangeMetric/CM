# 将原始数据汇总
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
import argparse

# calculates cohen's kappa value
def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2 + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return result


# calculates whether two accuracy arrays are statistically different according to GLM
def is_diff_sts(orig_accuracy_list, accuracy_list, threshold=0.05):
    len_list = len(orig_accuracy_list)

    zeros_list = [0] * len_list
    ones_list = [1] * len_list
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list

    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data)

    response, predictors = patsy.dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)

    # by default, is_kill = 0, meaning that the case is not statistically different from the original distribution
    is_kill = 0
    try:
        glm_results = glm.fit()
    except Exception as e:
        print(e)
        return is_kill

    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value = float(pv)

    effect_size = cohen_d(orig_accuracy_list, accuracy_list)
    is_kill = int((p_value < threshold) and effect_size >= 0.2)
    return is_kill


OPERATORS = ["mean", "std", "skew", "median", "var", "sem", "max", "min"]

def extract_feature(df: pd.DataFrame):
    feature_dict = {}

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

def dict2csv(dataset_summary_dict, output_dir, ft_num_all):
    df_first_train = pd.DataFrame(dataset_summary_dict['first_train'][0],index=[0])
    df_first_train.insert(0,'iter', 0)
    df_first_train.insert(0,'mutant', 'first_train')


    for i in range(10):
        df_ft = pd.DataFrame(dataset_summary_dict['ft'][i],index=[0])
        df_ft.insert(0,'iter', i)
        df_ft.insert(0,'mutant', 'ft')        
        if i == 0:
            final_df = pd.concat([df_first_train,df_ft],axis=0)
        else:
            final_df = pd.concat([final_df,df_ft],axis=0)
    for i in ft_num_all:
        for j in range(10):
            df = pd.DataFrame(dataset_summary_dict[f'ft_{i}'][j],index=[0])
            df.insert(0,'iter', j)
            df.insert(0,'mutant', f'ft_{i}')        
            final_df = pd.concat([final_df,df],axis=0)    
    final_df.to_csv(output_dir,index=False)


def extract_results(dataset_summary_dict, dir_name, model_dir):
    dataset_summary_dict[dir_name] = {}
    # log_dir = f"{model_dir}/results/{dir_name}/log.csv"
    if dir_name == "first_train":
        iter_num = 1
    else:
        iter_num = 10
    for i in range(iter_num):
        dataset_summary_dict[dir_name][i] = {}
        if dir_name == "first_train":
            feature_dir = f"{model_dir}/results/{dir_name}/monitor_features.csv"
        else:
            feature_dir = f"{model_dir}/results/{dir_name}/{i}/monitor_features.csv"

        # ---- log.csv ---- 
        # val_loss,val_accuracy,loss,accuracy
        # df = pd.read_csv(log_dir)
        # feature_dict = extract_feature(df)
        # for feat_key, feat_val in feature_dict.items():
        #     dataset_summary_dict[dir_name][feat_key] = feat_val

        # ---- monitor_detection.csv ----
        df = pd.read_csv(feature_dir)
        df = df.fillna(0.0)
        df = df.replace('False', 0)
        df = df.astype('float')
        df.drop(['not_converge', 'unstable_loss'], axis=1, inplace=True)
        feature_dict = extract_feature(df)
        for feat_key, feat_val in feature_dict.items():
            dataset_summary_dict[dir_name][i][feat_key] = feat_val

datasets = ['blob','circle','mnist','cifar10','reuters','imdb']


# 计算统计度量
def summary(base):
    for dataset in datasets:
        dataset_dir = f"{base}/{dataset}"
        for model in os.listdir(dataset_dir):

            dataset_summary_dict = {}
            model_dir = f"{dataset_dir}/{model}"
            if os.path.isfile(model_dir):
                continue
            print(f"{dataset}, {model}")
            if base == 'Evaluation':
                extract_results(dataset_summary_dict, 'first_train', f"Evaluation/{dataset}/{model}")
                extract_results(dataset_summary_dict, 'ft', f"Evaluation/{dataset}/{model}")
            if base == 'Evaluation':
                ft_num_all= [0,1,2,3,4]
            else:
                if dataset in ['reuters','imdb']:
                    ft_num_all= [10,11,20,21,30,31,40,41,50,51]
                else:
                    ft_num_all= [10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44,50,51,52,53,54]
            for ft_num in ft_num_all:
                extract_results(dataset_summary_dict, f"ft_{ft_num}", model_dir)
            dict2csv(dataset_summary_dict, f"{model_dir}/results/summary.csv", ft_num_all)


# 计算标签
def calculate_is_diff(base):
    for dataset in datasets:
        results_list = []
        model_num = 1
        for model_dir in os.listdir(f"{base}/{dataset}"):
            if not os.path.isdir(f"{base}/{dataset}/{model_dir}"):
                continue
            for i in range(10):
                results_list.append([model_dir, 'ft', i, 0])
            print(f"{model_num}, {dataset}, {model_dir}")
            model_num+=1
            results_dir = f"{base}/{dataset}/{model_dir}/results"
            orig_accuracy_list = []
            for i in range(10):
                data = pd.read_csv(f"Evaluation/{dataset}/{model_dir}/results/ft/{i}/log.csv")
                orig_accuracy_list.append(list(data['val_accuracy'])[-1])
            ave_origin_acc = sum(orig_accuracy_list) / len(orig_accuracy_list)
            if base == 'Evaluation':
                ft_num_all= [0,1,2,3,4]
            else:
                if dataset in ['reuters','imdb']:
                    ft_num_all= [10,11,20,21,30,31,40,41,50,51]
                else:
                    ft_num_all= [10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44,50,51,52,53,54]
            for ft_num in ft_num_all:
                accuracy_list = []
                for i in range(10):
                    data = pd.read_csv(f"{results_dir}/ft_{ft_num}/{i}/log.csv")
                    accuracy_list.append(list(data['val_accuracy'])[-1])     
                is_kill = is_diff_sts(orig_accuracy_list, accuracy_list)    # 是否有显著性差异 
                ave_cur_acc = sum(accuracy_list) / len(accuracy_list)
                is_faulty = int(is_kill and ave_cur_acc < ave_origin_acc)   # 
                if is_faulty:
                    for i in range(10):
                        if base == 'Evaluation':
                            results_list_tmp = [model_dir, f"ft_{ft_num}", i, 1]
                        else:
                            results_list_tmp = [model_dir, f"ft_{ft_num}", i, ft_num//10]
                        results_list.append(results_list_tmp)
                else:
                    for i in range(10):
                        results_list_tmp = [model_dir, f"ft_{ft_num}", i, 0]
                        results_list.append(results_list_tmp)
        test = pd.DataFrame(columns = ['model','mutant','iter','label'],data=results_list)
        test.to_csv(f'{base}/{dataset}/label.csv',index=False)
        # print(test)


# 计算初次训练和微调的度量差，并与标签合并
def get_diff_features(base):
    for dataset in datasets:
        dataset_dir = f"{base}/{dataset}"
        results=[]
        label = pd.read_csv(f"{base}/{dataset}/label.csv")
        model_num = 1
        for model in os.listdir(dataset_dir):
            if os.path.isfile(f"{base}/{dataset}/{model}"):
                continue
            print(f"{model_num}, {dataset}, {model}")
            model_num+=1
            model_dir = f"{dataset_dir}/{model}"
            df = pd.read_csv(f"{model_dir}/results/summary.csv",index_col=0)
            
            x = df.loc['first_train'] - df.loc[f'ft'] 
            x['iter'] = abs(x['iter']).astype("int")
            x['label'] = list(label[(label['model'] == model) & (label['mutant'] == f'ft')]['label'])[0]            
            x['model'] = model
            x['mutant'] = 'ft'
            results.append(x)

            if base == 'Evaluation':
                ft_num_all= [0,1,2,3,4]
            else:
                if dataset in ['reuters','imdb']:
                    ft_num_all= [10,11,20,21,30,31,40,41,50,51]
                else:
                    ft_num_all= [10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44,50,51,52,53,54]

            for i in ft_num_all:
                x = df.loc['first_train'] - df.loc[f'ft_{i}'] 
                x['iter'] = abs(x['iter']).astype("int")
                x['label'] = list(label[(label['model'] == model) & (label['mutant'] == f'ft_{i}')]['label'])[0]
                x['model'] = model
                x['mutant'] = f'ft_{i}'
                results.append(x)

        data = results[0]
        for i in range(1,len(results)):
            data = pd.concat([data,results[i]],axis=0)

        # data = pd.DataFrame(results)
        data.rename(columns={'model':'tmp1','mutant':'tmp2'},inplace = True)
        data.insert(0,'model',data['tmp1'])
        data.insert(1,'mutant',data['tmp2'])
        data.drop(axis=1,columns=['tmp1','tmp2'],inplace=True)
        data.to_csv(f"{dataset_dir}/features_diff.csv",index=False)

# 获取微调的过程度量
def get_old_metrics(base):
    for dataset in datasets:
        dataset_dir = f"{base}/{dataset}"
        results=[]
        label = pd.read_csv(f"{base}/{dataset}/label.csv")
        for model in os.listdir(dataset_dir):
            if os.path.isfile(f"{base}/{dataset}/{model}"):
                continue
            print(f"{dataset}, {model}")
            model_dir = f"{dataset_dir}/{model}"
            df = pd.read_csv(f"{model_dir}/results/summary.csv",index_col=0)
           
            x = df.loc[f'ft'].copy()
            x['label'] = list(label[(label['model'] == model) & (label['mutant'] == f'ft')]['label'])[0]
            x['model'] = model
            x['mutant'] = 'ft'
            results.append(x)

            if base == 'Evaluation':
                ft_num_all= [0,1,2,3,4]
            else:
                if dataset in ['reuters','imdb']:
                    ft_num_all= [10,11,20,21,30,31,40,41,50,51]
                else:
                    ft_num_all= [10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44,50,51,52,53,54]
            for i in ft_num_all:
                x = df.loc[f'ft_{i}'].copy()
                x['label'] = list(label[(label['model'] == model) & (label['mutant'] == f'ft_{i}')]['label'])[0]
                x['model'] = model
                x['mutant'] = f'ft_{i}'
                results.append(x)

        data = results[0]
        for i in range(1,len(results)):
            data = pd.concat([data,results[i]],axis=0)

        # data = pd.DataFrame(results)
        data.rename(columns={'model':'tmp1','mutant':'tmp2'},inplace = True)
        data.insert(0,'model',data['tmp1'])
        data.insert(1,'mutant',data['tmp2'])
        data.drop(axis=1,columns=['tmp1','tmp2'],inplace=True)
        data.to_csv(f"{dataset_dir}/features_old.csv",index=False)    


def preprocess_summary2(model_dir, dir_name):
    for i in range(10):
        with open(f"{model_dir}/results/{dir_name}/{i}/monitor_features.csv",'r') as f:
            lines = f.readlines()
            lines[0] = lines[0].replace("\n","").replace(",loss,accuracy,val_loss,val_accuracy","")+",loss,accuracy,val_loss,val_accuracy\n"
        with open(f"{model_dir}/results/{dir_name}/{i}/monitor_features.csv",'w') as f:
            for line in lines:
                f.write(line)

# 加列名
def preprocess_summary(base):
    for dataset in datasets:
        dataset_dir = f"{base}/{dataset}"
        for model in os.listdir(dataset_dir):
            model_dir = f"{dataset_dir}/{model}"
            if os.path.isfile(model_dir):
                continue
            print(f"{dataset},{model}")
            # preprocess_summary2(model_dir,"first_train")
            # preprocess_summary2(model_dir,"ft")
            if base == 'Evaluation':
                ft_num_all= [0,1,2,3,4]
            else:
                if dataset in ['reuters','imdb']:
                    ft_num_all= [10,11,20,21,30,31,40,41,50,51]
                else:
                    ft_num_all= [10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44,50,51,52,53,54]
            for i in ft_num_all:
                preprocess_summary2(model_dir,f"ft_{i}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', '-bs', default='Evaluation', choices=['Evaluation','Evaluation_sp'], help='Base directory. Default: Evaluation')
    args = parser.parse_args()

    preprocess_summary(args.base)
    summary(args.base)
    calculate_is_diff(args.base)
    get_diff_features(args.base)
    get_old_metrics(args.base)

