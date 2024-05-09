import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scp
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import flowers102
import skimage.io as skio

#Hyperparameters
Batch_Size_Hyperparam = 4
Epoch_hyperparam = 100
img_H = 256
img_W = 256

class my_NN(nn.Module):
    def __init__(self):
      super(my_NN,self).__init__()
    
      self.layer1 = nn.Sequential(
         nn.Conv2d(3,32,kernel_size=3, padding=0,stride=2),
         nn.BatchNorm2d(32),
         nn.ReLU(),
         nn.MaxPool2d(2)
       )
    
      self.layer2 = nn.Sequential(
          nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
    
      self.layer3 = nn.Sequential(
          nn.Conv2d(64,128, kernel_size=3, padding=0, stride=2),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.MaxPool2d(2)
       )
    
    
      self.fc1 = nn.Linear(1152,1024)
      self.fc2 = nn.Linear(1024,1024)
      self.fc3 = nn.Linear(1024,512)
      self.fc4 = nn.Linear(512,102)
      self.relu = nn.ReLU()
    
    
    def forward(self,x):
    
       out1 = self.layer1(x)
       out1 = self.layer2(out1)
       out1 = self.layer3(out1)
       out1 = out1.view(out1.size(0),-1)
       
       out1 = self.relu(self.fc1(out1))
       out1 = self.relu(self.fc2(out1))
       out1 = self.relu(self.fc3(out1))
       out1 = self.fc4(out1)
       return out1

def main() -> None:

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(device)

  model=my_NN()
  model=model.to(device)
  print(model)


  optimizer = optim.Adam(             # Optimiser
      model.parameters(),
      lr = 0.001
      )

  criterion = nn.CrossEntropyLoss()   # Loss function



  normalise = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

  train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((img_H, img_W)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomSolarize(5, p=0.3),
        #transforms.RandomGrayscale(p=0.3),
        transforms.ToTensor(),
        normalise
    ])
  val_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((img_H, img_W)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalise
    ])

  test_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((img_H, img_W)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalise
    ])

  # Splitting dataset into train, test and val

  train_set = flowers102.Flowers102(root="data", split="train", download=True, transform = train_transform)
  val_set = flowers102.Flowers102(root="data", split="val",download=True, transform = val_transform)
  test_set = flowers102.Flowers102(root="data", split="test",download=True, transform = test_transform)

  # dataloaders
  train_loader = DataLoader(train_set, batch_size=Batch_Size_Hyperparam, shuffle=True, num_workers=4)
  test_loader = DataLoader(test_set, batch_size=Batch_Size_Hyperparam, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_set, batch_size=Batch_Size_Hyperparam, shuffle=True, num_workers=4)

  #Validation
  def check_val_accuracy(val_loader, model):
      model.eval()
      val_loss, correct = 0,0
      num_batches = len(val_loader)
      with torch.no_grad():
        for x,y in val_loader:
          # sending the data to the device
          x = x.to(device)
          y = y.to(device)

          pred = model(x)
          val_loss += criterion(pred, y).item()
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                  
          val_loss /= num_batches    
          #prints error and accuracy every after every epoch 
        print(f"Accuracy on the validation images: {100 * correct/1020:>0.1f}%, Avg loss: {val_loss:>5f} \n ")
      model.train()

  #Testing
  def check_test_accuracy(test_loader, model):
      model.eval()
      test_loss, correct = 0,0
      num_batches = len(test_loader)
      with torch.no_grad():
        for x,y in test_loader:
          # sending the data to the device
          x = x.to(device)
          y = y.to(device)

          pred = model(x)
          test_loss += criterion(pred, y).item()
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                  
          test_loss /= num_batches    

        print(f"Accuracy on the Test images: {100 * correct/6149:>0.1f}%, Avg loss: {test_loss:>5f} \n ")
      model.train()
  

  for epoch in range(Epoch_hyperparam):  # loop over the dataset multiple times 
      running_loss = 0.0
      for i, data in enumerate(train_loader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs = inputs.to(device)
          labels = labels.to(device)

          # zero the parameter gradients
          outputs = model(inputs)
          loss=criterion(outputs, labels) 
          optimizer.zero_grad()

          # forward + backward + optimize

          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
        #   if i % 200 == 199:    # print every 200 mini-batches
        #       print('[%d, %5d] loss: %.3f' %
        #             (epoch + 1, i + 1, running_loss / 2000))
            #   running_loss = 0.0
      print(f"Epoch {epoch}: Average loss: {running_loss/1020}")
      check_val_accuracy(val_loader, model)
  
  check_test_accuracy(test_loader, model)

  torch.save(model.state_dict(), 'flower_classification_model.pth')




if __name__ == "__main__":
    main()

