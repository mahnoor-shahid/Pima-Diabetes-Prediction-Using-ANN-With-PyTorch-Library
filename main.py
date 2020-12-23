########## loading necessary libraries ##########
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import classes.model as ann
import classes.dataLoader as dL

########## main program ##########
if __name__ == '__main__':

    ########## fetching dataset from the loader ##########
    dataset = dL.datasetLoader(path = '.\\dataset\\diabetes.csv')

    ########## setting up training data ##########
    train_loader = DataLoader(dataset=dataset,
                            batch_size=64,
                            shuffle=True)

    ########## Instantiate ANN model ##########
    model = ann.ANN_model()
    print(model.parameters)

    ########## Loss Function and Optimizer ##########
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    ########## Training Loop ##########
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):

            # get the inputs
            inputs, labels = data
            y_pred = model.forward(inputs)

            loss = loss_function(y_pred, torch.max(labels,1)[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch Number {} and the loss is : {} ".format(epoch, loss.item()))
