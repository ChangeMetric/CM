import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def count_label():
    all_data = {}
    for dataset_name in ['circle','blob','mnist','cifar10','reuters','imdb']:
        label_data = {}
        for prop in [10,20,30,40,60,80]:
            df = pd.read_csv(f"../data2/{dataset_name}/label_{prop}.csv")
            x = Counter(df['label'])
            label_data[prop] = x
        df = pd.read_csv(f"../data2/{dataset_name}/label.csv")
        x = Counter(df['label'])
        label_data[50] = x    

        new_data = []
        for prop in [10,20,30,40]:
            new_data.append([int(label_data[prop][1]/5), int(label_data[prop][2]/5) ,int(label_data[prop][3]/5),int(label_data[prop][4]/5),int(label_data[prop*2][5]/5)])
        new_data.append([int(label_data[50][1]/5), int(label_data[50][2]/5),int(label_data[50][3]/5),int(label_data[50][4]/5),int(label_data[50][5]/5)])
        data_df = pd.DataFrame(new_data,columns=['NP','LE','DR','DM','DS'])
        all_data[dataset_name] = data_df
    all_data_df = pd.DataFrame(np.zeros_like(all_data['blob']),columns=all_data['blob'].columns)
    for dataset_name in ['circle','blob','mnist','cifar10','reuters','imdb']:
        all_data_df += all_data[dataset_name]
    # print(all_data_df)

    for i in all_data_df.columns:
        sns.lineplot(x=range(5), y=all_data_df[i], label=i,marker='o')
    plt.xticks(range(5),['10%','20%','30%','40%','50%'])
    plt.xlabel("Mutation Proportion (for NP, LE, DR and DM)")
    plt.ylabel("Faulty Model")
    plt.ylim(0, 200)

    ax1 = plt.gca()  # 获取当前轴
    ax2 = ax1.secondary_xaxis('top')  # 创建第二个横坐标轴
    ax2.set_xticks(range(5))  # 设置刻度位置
    ax2.set_xticklabels(['20%','40%','60%','80%','100%'])  # 设置上方标签
    ax2.set_xlabel("Mutation Proportion (for DS)")  # 设置顶部横坐标标签

    plt.legend(title="Mutation Operator", ncol=2)
    plt.show()

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    count_label()