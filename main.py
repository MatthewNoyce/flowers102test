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

def download() -> flowers102.Flowers102:
     return flowers102.Flowers102("./myData", download=True)

class my_NN(nn.Module):
    def __init__(self):
      super(my_NN,self).__init__()
    
      self.layer1 = nn.Sequential(
         nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
         nn.BatchNorm2d(16),
         nn.ReLU(),
         nn.MaxPool2d(2)
       )
    
      self.layer2 = nn.Sequential(
          nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
    
      self.layer3 = nn.Sequential(
          nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d(2)
       )
    
    
      self.fc1 = nn.Linear(576,512)
      self.fc2 = nn.Linear(512,512)
      self.fc3 = nn.Linear(512,102)
      self.relu = nn.ReLU()
    
    
    def forward(self,x):
    
       out1 = self.layer1(x)
       out1 = self.layer2(out1)
       out1 = self.layer3(out1)
       out1 = out1.view(out1.size(0),-1)
       
       out1 = self.relu(self.fc1(out1))
       #out1 = self.relu(self.fc5(out1))
       out1 = self.relu(self.fc2(out1))
       out1 = self.fc3(out1)
       return out1

class MyFlowerDataset(Dataset):
  def __init__(self, metadata, transform=None):
    self.metadata = metadata
    self.transform = transform

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    image_path = self.metadata.iloc[idx, 0]
    image = skio.imread(image_path)
    label = torch.tensor(int(self.metadata.iloc[idx, 1]))
    # label = F.one_hot(label, num_classes=102)
    # label = label.float()
    if self.transform:
      image = self.transform(image)
    return (image, label)

flower_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def main() -> None:
  download()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(device)

  model=my_NN().to(device)
  print(model)


  optimizer = optim.Adam(             # Optimiser
      model.parameters(),
      lr = 0.001
      )

  criterion = nn.CrossEntropyLoss()   # Loss function

  data_path = os.path.join("myData","flowers-102","jpg")
  label_path = os.path.join("myData","flowers-102","imagelabels.mat")
  label_array = scp.loadmat(label_path)["labels"]
  label_array -= 1

  def check_accuracy(loader, model):
    num_corrects = 0
    num_samples = 0
    model.eval()
    test_loss, correct = 0,0
    num_batches = len(loader)
    with torch.no_grad():
      for x,y in loader:
        # sending the data to the device
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        test_loss += criterion(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                
        test_loss /= num_batches    
        #correct /= 1020
        #prints error and accuracy every after every epoch 
      print(f"Test Error: \n Accuracy: {100 * correct/1020:>0.1f}%, Avg loss: {test_loss:>5f} \n ")
    model.train()


  labels_list = list(label_array[0, :])
  image_path_list = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
  metadata = pd.DataFrame(
      {'image_path': image_path_list,
      'image_label': labels_list}
  )

  my_flowers = MyFlowerDataset(metadata, transform = flower_transform)

  # Splitting dataset into train, test and val
  train_set, test_set, val_set = torch.utils.data.random_split(my_flowers, [1020, 6149, 1020]) #changed for faster training on laptop
  


  # dataloaders
  train_loader = DataLoader(train_set, batch_size= 4, shuffle=True, num_workers=4)
  test_loader = DataLoader(test_set, batch_size= 4, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_set, batch_size= 4, shuffle=False, num_workers=4)

  for epoch in range(2):  # loop over the dataset multiple times
      check_accuracy(val_loader, model)
      running_loss = 0.0
      for i, data in enumerate(train_loader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs=inputs.to(device)
          labels=labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = model(inputs)
          loss=criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 200 == 199:    # print every 200 mini-batches
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
              running_loss = 0.0
      

if __name__ == "__main__":
    main()

