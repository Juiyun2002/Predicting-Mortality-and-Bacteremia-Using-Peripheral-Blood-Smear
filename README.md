# Predicting-Mortality-and-Bacteremia-Using-Peripheral-Blood-Smear

## Introduction
In this project, we investigate the potential of machine learning models to predict patient mortality (within 1, 3, 5, and 7 days) and bacteremia using structured clinical data and PBS images. By applying deep learning architectures (e.g., ResNet, DenseNet, MedGemma) and attention-based multiple instance learning, our models achieve promising performance, with AUC scores reaching up to 0.90 for mortality and 0.79 for bacteremia prediction. 

## Dataset
- The file `lab_data` contains structured clinical data with all personally identifiable information removed.
- The PBS image dataset used in this project is not publicly available.

## Train CNN models from PBS

- Run `resnet.py` or `densenet.py`, then the predicted 1, 3, 5, and 7-day mortality outcomes for the testing set will be saved.
- For example, the 1-day mortality predictions and corresponding ground truth from ResNet will be saved as `resnet_1d_pred.npy` and `resnet_1d_labels.npy`, respectively.

## Train MIL models from PBS and WBC

For mortality prediction, run `MIL_mortality.ipynb`
- Set `DAYS` to the number of days for predicting mortality
- Set `MAX_WBC` to the maximum number of WBC images per bag

For bacteremia prediction, run `MIL_bacteremia.ipynb`
- Set `MAX_WBC` to the maximum number of WBC images per bag

## Train Structured Data Models
To run the structured data models for bacteremia and mortality (1, 3, and 5 days) prediction:

Data Used:
  Structured clinical data in csv_data/lab_data 20240509-20240801.csv & csv_data/input_v5.csv &csv_data/test.csv
How to Run:
  python csv_bacteremia.py 
  python csv_mortality.py 

