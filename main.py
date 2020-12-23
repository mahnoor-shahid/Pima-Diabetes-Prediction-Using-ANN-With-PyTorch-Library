########## loading necessary libraries ##########
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
# import torch.nn as nn
import torch.nn.functional as F
import ANN_model as ann

########## main program ##########
if __name__ == '__main__':

    # loading dataset
    dataset = pd.read_csv('dataset\\diabetes.csv')
    # exploring the uploaded data frame
    print(dataset.head())

    # checking for the missing values 
    dataset.isnull().sum()

    # replacing outcome values with 'diabetic' and 'not diabetic'
    dataset['Outcome'] = np.where(dataset['Outcome']==1,'Diabetic', 'Not Diabetic')
    # checking the values of outcome
    print(np.unique(dataset['Outcome']))

    ########## splitting dataset into training and testing data ##########
    y = dataset['Outcome'].values ## dependent variable
    X = dataset.drop('Outcome', axis=1).values ## independent variables

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ########## Creating tensors ##########
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(X_train)
    y_test = torch.FloatTensor(X_train)

    a = ann()