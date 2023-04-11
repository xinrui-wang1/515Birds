# Bird Species Classification using ResNet-50

## Description
This project aims to classify 515 different bird species using ResNet, we will also explore model explainability through LIME. 

## Dataset
https://www.kaggle.com/datasets/gpiosenka/100-bird-species

The dataset used for this project is obtained from Kaggle, containing images of 515 different bird species. The dataset is pre-split into training (82724 samples), validation (2575 samples), and test (2575 samples).

## Model Used
The CNN ResNet50 architecture was used to classify the bird species. The model was pre-trained on the ImageNet dataset and then fine-tuned on the bird species dataset.

## Result
After 20 epochs of trainning using CorssEntropy Loss and Adam Optimizer, we have:

- train loss: 0.0501
- train accuracy: 0.9848
- validation loss: 0.4280
- validation accuracy: 0.8971

### Train/Validation Loss and Accuracy
![Train, Validation Loss and Accuracy](https://github.com/xinrui-wang1/bird-classification/blob/main/figures/train_vs_val_loss_accuracy.png)

### Sample Predictions
![Sample Predictions](https://github.com/xinrui-wang1/bird-classification/blob/main/figures/sample_predictions.png)

### Most Mispredicted Classes
![Most Mispredicted Classes](https://github.com/xinrui-wang1/bird-classification/blob/main/figures/top_mispredicted.png)

## Requirement
## Usage
## Reference
https://www.kaggle.com/datasets/gpiosenka/100-bird-species

https://arxiv.org/pdf/1512.03385.pdf
