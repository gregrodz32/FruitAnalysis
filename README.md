# FruitAnalysis

This repository trains a Convolutional Neural Network (CNN) to classify fruits.

1. Download the "Fruits 360" dataset from [Fruits 360 Kaggle Dataset](https://www.kaggle.com/datasets/moltean/fruits).
2. Extract the dataset and place it in the `dataset/` folder. It should have the following structure:
   dataset ├── Training/
           ├── Test/
           ├── test-results/
           ├── test-results-multiple-fruits
           ├── test-multiple-fruits/

3. Train the model by running the fruit_classifier.py (you can adjust the epoch and CNN variables)

4. set the epoch to the saved_model you would like to use on the test.py file 
