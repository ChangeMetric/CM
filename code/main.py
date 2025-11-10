import os
import time
import pickle
import keras
import numpy as np
import pandas as pd

from keras.models import load_model
import keras.optimizers as O
from keras.datasets import cifar10, mnist, imdb, reuters
import keras.backend as backend
from Callbacks.LossHistory import LossHistory


from sklearn.datasets import make_blobs

from get_dataset import get_dataset,get_ft_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_num = {'blob':39,'circle':36, 'cifar10':35, 'mnist':78, 'reuters':32,'imdb':13}

# datasets = ['blob','circle','cifar10','mnist','reuters','imdb']
datasets=['circle']

params = {'beta_1': 1e-3,
          'beta_2': 1e-4,
          'beta_3': 70,
          'gamma': 0.7,
          'zeta': 0.03,
          'eta': 0.2,
          'delta': 0.01,
          'alpha_1': 0,
          'alpha_2': 0,
          'alpha_3': 0,
          'Theta': 0.6
          }

def parse_train_config(config):
    opt_cls = getattr(O, config['optimizer'])
    opt = opt_cls(**config['opt_kwargs'])
    batch_size = config['batchsize']
    # add upper bound to batch size due to OOM
    if config['dataset'] == "reuters" or config['dataset'] == "imdb":
        batch_size = min(16, batch_size)
    epoch = config['epoch']
    loss = config['loss']
    callbacks = [] if 'callbacks' not in config.keys() else config['callbacks']
    return opt, batch_size, epoch, loss, callbacks

def my_train():
    for dataset_name in datasets:
        dataset = get_dataset(dataset_name)
        time_list = []
        for num in range(model_num[dataset_name]):
            model_path = f"../models/{dataset_name}/{num}/model.h5"
            config_path = f"../models/{dataset_name}/{num}/training_config.pkl"
            save_dir = f"../models/{dataset_name}/{num}/monitor_features/pre_train"
            model_structure = load_model(model_path)
            with open(config_path, 'rb') as f:
               training_config = pickle.load(f)
            opt, batch_size, epoch, loss, callbacks = parse_train_config(training_config)
            print(f"{dataset_name},{num}")
            print("opt: {} | batch: {} | epoch: {} | loss: {} | callbacks: {}".format(opt, batch_size, epoch, loss, callbacks))
            model_structure.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
            callbacks = []
            # callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3,
            #                                            verbose=1, mode='auto', baseline=None,
            #                                            restore_best_weights=False))
            callbacks.append(LossHistory(training_data=[dataset['x'], dataset['y']], model=model_structure, determine_threshold=1,
                    batch_size=batch_size, save_dir=save_dir, total_epoch=epoch, satisfied_acc=0.7,
                    checktype='epoch_1', params=params))
            stime = time.time()
            model_structure.fit(dataset['x'], dataset['y'], batch_size=batch_size, validation_split=0.25, epochs=epoch, verbose=0, callbacks=callbacks)
            model_structure.save(f"{save_dir}/trained_model.h5")
            etime = time.time()
            time_list.append([dataset_name,num,etime-stime])
            
            with open(f"{save_dir}/test_results.txt","w",encoding="utf-8") as f2:
                test_loss, test_accuracy = model_structure.evaluate(dataset['x_val'], dataset['y_val'])
                print(test_loss,test_accuracy)
                f2.write(f"{test_loss},{test_accuracy}\n")
            backend.clear_session()
        df=pd.DataFrame(time_list,columns=['dataset','num','time'])
        df.to_csv(f"../results/time_{dataset_name}_pretrain.csv",index=False)

def my_fine_tune(no):
    for dataset_name in datasets:
        dataset = get_ft_dataset(dataset_name, no)
        time_list = []
        for num in range(model_num[dataset_name]):
            model_path = f"../models/{dataset_name}/{num}/monitor_features/pre_train/trained_model.h5"
            config_path = f"../models/{dataset_name}/{num}/training_config.pkl"
            with open(config_path, 'rb') as f:
                training_config = pickle.load(f)
            opt, batch_size, epoch, loss, callbacks = parse_train_config(training_config)
            print("opt: {} | batch: {} | epoch: {} | loss: {} | callbacks: {}".format(opt, batch_size, epoch, loss, callbacks))
            for iter in range(5):
                model_structure = load_model(model_path)
                model_structure.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
                print(f"{dataset_name},{num},{iter}")
                if not os.path.exists(f"../models/{dataset_name}/{num}/monitor_features/fine_tune_{no}/{iter}"):
                    os.makedirs(f"../models/{dataset_name}/{num}/monitor_features/fine_tune_{no}/{iter}")
                save_dir = f"../models/{dataset_name}/{num}/monitor_features/fine_tune_{no}/{iter}"
                callbacks = []
                callbacks.append(LossHistory(training_data=[dataset['x'], dataset['y']], model=model_structure, determine_threshold=1,
                        batch_size=batch_size, save_dir=save_dir, total_epoch=epoch, satisfied_acc=0.7,
                        checktype='epoch_1', params=params))
                stime = time.time()
                model_structure.fit(dataset['x'], dataset['y'], batch_size=batch_size, validation_split=0.25, epochs=5, verbose=1, callbacks=callbacks)
                model_structure.save(f"{save_dir}/trained_model.h5")
                etime = time.time()
                print(etime-stime)
                time_list.append([dataset_name,num,iter,etime-stime])
                with open(f"{save_dir}/test_results.txt","w",encoding="utf-8") as f2:
                    test_loss, test_accuracy = model_structure.evaluate(dataset['x_val'], dataset['y_val'])
                    print(test_loss,test_accuracy)
                    f2.write(f"{test_loss},{test_accuracy}\n")
                backend.clear_session()
        df=pd.DataFrame(time_list,columns=['dataset','num','iter','time'])
        df.to_csv(f"../results/time_{dataset_name}_ft_{no}.csv",index=False)



if __name__=="__main__":
    # my_train()
    for i in range(0,6):
        my_fine_tune(i)


