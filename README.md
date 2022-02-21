# AlexNet
Train AlexNet using the architecture described in the paper "[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)"

## Requirements
* 64-bit Python 3.7 (Anaconda environment is recommended)
* Tensorflow 2.8.0

## Getting started
1. Obtaining dataset\
The available dataset at this moment is the ILSVRC 2012-2017 image classification and localization dataset which could be downloaded through [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).\
`kaggle competitions download -c imagenet-object-localization-challenge`\
Note: If you have not used Kaggle before please follow this [link](https://www.kaggle.com/docs/api) to properly set up Kaggle. 

2. Training AlexNet from scratch
    - Execute preprocess.py file to perform dataset preprocessing

    - Execute train.py file
        - `--outdir` Setting the output file path (str)
        - `--data`   Pointing the training dataset path (str)
        - `--resume` Whether resuming from the latest checkpoint (boolean)
        - `--epoch` Determine how many epochs to train (int)



