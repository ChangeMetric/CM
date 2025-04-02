# Change Metrics

## Overview
This repository stores our experimental codes, datasets and results.
```
├─code                          # The source code of our approach 
├─metrics                       # Metrics extracted from runtime data, used to train classification models
├─models                        # The DL models used in our experiments
└─results                       # The results of our experiments
    ├─evaluation                # Evaluations of the classification models
    ├─gain_ratio                # The gain_ratio of each metric
    ├─pred                      # Predictions results of the classification models
    └─time                      # Training time
```

## Datasets

### DL models
In this experiment, we use 233 DNN models. Each model contains the initial model structure (`model.h5`) and hyper-parameter (`training_config.pkl`), which is stored under the `models` folders. 

### Metrics Calculation
The metrics used to train classification models are stored in the `metrics` folder. There are three categories of metrics used in the experiments.
 - **Baseline**: The baseline metrics are calculated by 8 statistical operators based on the runtime data. The results are stored in the file `fine_tune_runtime.csv`.
 - **Descriptive Statistics Difference Metrics(DSDM)**: The metrics for this category are calculated by subtracting the statistics of the pretraining runtime data from the statistics of the fine-tuning runtime data.  The results are stored in the file `diff_features.csv`.
 - **Multidimensional Data Comparison Metrics(MDCM)**: The metrics for this category are calculated by comparing the differences of each kind of runtime data collected during the pretraining and the fine-tuning processes from several aspects. The results are stored in the file `diff_features2.csv`.

## Evaluation
We use five classification algorithms to train classifiers and use three indicators to evaluate the performance of the classifiers. The results are under the `results` folder. Our results can be reproduced as follows.

### Requirement
- Python 3.8
- Packages:
```
pip install -r requirements.txt
```

### RQ1
- Training classifiers to diagnosis what kind of errors is present in the data and evaluating performence 
```
python train.py -o train
```
- Comparing the results between DSDM+MDCM and baseline
```
python train.py -o compare -m1 BC -m2 A
```

### RQ2
- Comparing the results between DSDM and DSDM+MDCM
- Comparing the results between MDCM and DSDM+MDCM
```
python train.py -o compare -m1 B -m2 BC
python train.py -o compare -m1 C -m2 BC
```

### RQ3
- Count the time taken by our approach
```
python get_time.py
```

### Discussion
- Calculate the gain ratio of individual metric
```
python gain_ratio.py
```
- Effectiveness of fault diagnosis for different classes
```
python train.py -o each_class
```
- Compare the performance of RF classifier with and without PCA process.
```
python pca_process.py
```
- The Impact of Different Mutation Propotion
```
python count_labels.py
```