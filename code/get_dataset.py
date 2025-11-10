import pickle
import keras
from keras.preprocessing import sequence
from keras.datasets import cifar10, mnist, imdb, reuters

def get_dataset(dataset_name):
    dataset = {}
    if dataset_name == "blob":
        with open("../data/data_blob.pkl","rb") as f:
            dataset = pickle.load(f)
    elif dataset_name == "circle":
        with open("../data/data_circle.pkl","rb") as f:
            dataset = pickle.load(f)        
    elif dataset_name == "mnist":
        (x, y), (x_val, y_val) = mnist.load_data()
        dataset['x']=x.reshape(60000,28,28,1).astype('float32')/255
        dataset['x_val']=x_val.reshape(10000,28,28,1).astype('float32')/255
        dataset['y'] = keras.utils.to_categorical(y, 10)
        dataset['y_val'] = keras.utils.to_categorical(y_val, 10)
    elif dataset_name == "cifar10":
        (x, y), (x_val, y_val) = cifar10.load_data()
        dataset['x'] = x.astype('float32')/255
        dataset['x_val'] = x_val.astype('float32')/255
        dataset['y'] = keras.utils.to_categorical(y, 10)
        dataset['y_val'] = keras.utils.to_categorical(y_val, 10)
    elif dataset_name == "imdb":
        (x, y), (x_val, y_val) = imdb.load_data(num_words=10000)
        dataset['x']=sequence.pad_sequences(x, maxlen=300)
        dataset['x_val']=sequence.pad_sequences(x_val, maxlen=300)
        dataset['y'] = keras.utils.to_categorical(y, 2)
        dataset['y_val'] = keras.utils.to_categorical(y_val, 2)
    elif dataset_name == 'reuters':
        with open("../data/data_reuters.pkl","rb") as f:
            dataset = pickle.load(f)       
    return dataset


def get_ft_dataset(dataset_name, no):
    dataset = {}
    if no == 0:
        with open(f"../data/new_data/{dataset_name}/data_{dataset_name}_ft.pkl","rb") as f:
            dataset = pickle.load(f)
    else:
        with open(f"../data/new_data/{dataset_name}/data_{dataset_name}_ft_{no}.pkl","rb") as f:
            dataset = pickle.load(f)    
    return dataset    