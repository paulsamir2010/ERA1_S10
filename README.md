# ERA1_S10
ERA1 Program from "The School of AI" (TSAI) -- Session 10 - Residual Connections in CNNs and One Cycle Policy

## Objective and Requirement

Dataset = CIFAR10 Used Pytorch Framework

Requirement is to use batch size of 512, use One Cycle LR policy, with a custom ResNet kind of Model (specification given below)

Target is to achieve 90% test accuracy within 24 epochs

  ### Model Specification given below:

  PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]

  #### Layer1 -

    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]

    R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]

    Add(X, R1)

  #### Layer 2 -

    Conv 3x3 [256k]

    MaxPooling2D

    BN

    ReLU

  #### Layer 3 -

    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]

    R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]

    Add(X, R2)

    MaxPooling with Kernel Size 4

    FC Layer

    SoftMax

### Uses One Cycle Policy such that:

     Total Epochs = 24

     Max at Epoch = 5

     LRMIN = FIND

     LRMAX = FIND

     NO Annihilation

## Model Specification

![image](https://github.com/paulsamir2010/ERA1_S10/blob/main/ModelSummary.jpg)

## Organization of files in this repository

- Model is in mymodels.py

- Train and test code are in mytrain.py and mytest.py

- Main ipynb file is base_main.ipynb

## Maximum LR based on Steepest Gradient

![image](https://github.com/paulsamir2010/ERA1_S10/blob/main/MaxLR.jpg)

## One Cycle Learning Rate

![image](https://github.com/paulsamir2010/ERA1_S10/blob/main/OnecycleLR.jpg)

## Training Losses and Train Accuracy

![image](https://github.com/paulsamir2010/ERA1_S10/blob/main/TrainingLossAcc.jpg)

## Misclassified Images

![image](https://github.com/paulsamir2010/ERA1_S10/blob/main/Misclassified.jpg)

## Results

Test Accuracy obtained = 93.00%   within 24 EPOCHS
