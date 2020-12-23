########## loading necessary libraries ##########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    final_losses = []
    epochs = 10
    for e in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model.forward(inputs)

            loss = loss_function(y_pred, torch.max(labels,1)[1])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_losses.append(loss)
        print("Epoch Number {} and the loss is : {} ".format(e, loss.item()))
    
    ########## Plot the loss function ##########
    plt.plot(range(epochs),final_losses, color='red', lw="2", ls="solid", marker="o", markerfacecolor="purple", markersize="6", alpha=0.5)
    plt.title('Loss per every epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.show()
