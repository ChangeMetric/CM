import pandas as pd
import numpy as np

datasets=['blob','circle','mnist','cifar10','reuters','imdb']
model_num = {'blob':39,'circle':36, 'cifar10':35, 'mnist':78, 'reuters':32,'imdb':13}

def get_time():
    for dataset_name in datasets:
        df_pretrain = pd.read_csv(f"../results/time/pretrain/time_{dataset_name}_pretrain.csv")
        pretrain_time = np.mean(df_pretrain['time'])
        fine_tune_time = 0
        for i in range(6):
            df_ft = pd.read_csv(f"../results/time/fine-tune/time_{dataset_name}_ft_{i}.csv")
            fine_tune_time+=np.sum(df_ft['time'])
            # print(np.mean(df_ft['time']))
        fine_tune_time /= (model_num[dataset_name]*5*6)
        with open(f"../metrics/{dataset_name}/time.txt","r",encoding='utf-8') as f:
            cm_time = float(f.read())
            cm_time /= (model_num[dataset_name]*5*6)
        with open(f"../results/pred/{dataset_name}/time.txt","r",encoding='utf-8') as f:
            pred_time = float(f.read())
            pred_time /= 50
        print(f"{pretrain_time}\t{fine_tune_time}\t{cm_time}\t{pred_time}")

 
if __name__ == "__main__":
    get_time()