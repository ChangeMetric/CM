# CM

This repository stores our experimental codes and results.

## Requirement
- Python 3.8
- Packages:
```
pip install -r requirements.txt
```

## Model
In this experiment, we use 233 DNN models. Each model contains model structure(model.h5) and hyper-parameter(training_config.pkl), which is stored under the `models` folders. 

## Runtime Data
The runtime data are under the folder `runtime_data`. Due to space limitation, the folder contains the runtime data of two models as examples, one from 'blob', and the other from 'circle'. The results contains many folders:
- `first_train`: runtime data of initial training
- `ft`: runtime data of retraining with benchmark data
- `ft_{num}`: runtime data of retraining with mutated data


## Metrics Calculation
There are three categories of metrics used in the experiments.

 - **Baseline**: the baseline is calculated by 8 statistical operators based on the runtime data. The code to calculate these metrics is in `preprocess.py`. The results are stored in the file `features_old.csv`.
 - **Descriptive Statistics Difference Metrics(DSDM)**: The metric for this category is calculated by subtracting the statistics of the pre-training runtime data from the statistics of the fine-tuning runtime data. The code to calculate these metrics is in `preprocess.py`. The results are stored in the file `features_diff.csv`.
 - **Multidimensional Data Comparison Metrics(MDCM)**: The metric for this category is calculated by comparing the differences of each kind of runtime data sequence collected during the pre-training process and the fine-tuning process from several aspects.The code to calculate these metrics is in `change_metrics.py`. The results are stored in the file `change_features.csv`.

## Evaluation
We use five classification algorithms to train classifiers and use 10 times 10-fold cross-validation to evaluate the performance of the classifiers. The experiment can be reproduced by executing `train.py` and `train_sp.py`. The results are under the `results` folder of `Evaluation` and `Evaluation_sp`.

### RQ1
- Training classifiers to detect whether there are errors in the fine-tuning data
- Evaluating performence 
- Comparing the results between DSDM+MDCM and baseline
```
python train.py  -t 1 -a 1 -e 1 -m1 BC -m2 A
```

### RQ2
- Comparing the results between DSDM and DSDM+MDCM
- Comparing the results between MDCM and DSDM+MDCM
```
python train.py  m1 B -m2 BC
python train.py  m1 C -m2 BC
```

### RQ3
- Training classifiers to diagnosis what kind of errors is present in the data
- Evaluating performence 
- Comparing the results between DSDM+MDCM and baseline
```
python train_sp.py  -t 1 -a 1 -m1 BC -m2 A
```

### RQ4
- Comparing the results between DSDM and DSDM+MDCM
- Comparing the results between MDCM and DSDM+MDCM
```
python train_sp.py  m1 B -m2 BC
python train_sp.py  m1 C -m2 BC
```

## Discussion
The figures used in discussion section can be generated by the code in `discussion.py`.
```
python disscussion.py -d 1
python disscussion.py -d 2
python disscussion.py -d 3
python disscussion.py -d 4
```