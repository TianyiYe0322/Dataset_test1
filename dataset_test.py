import torch

#from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
torch.manual_seed(0)

from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os

directory = '/Users/tianyiye/Documents/python_project/Dataset_test/2243_9243_bundle_archive'
csv_file = 'fashion-mnist_test.csv'

csv_path=os.path.join(directory,csv_file)

data_name = pd.read_csv(csv_path)
data_name.head()

data_train = 'train-images-idx3-ubyte'
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                                shuffle = True)
#print (data_loader_train)
#transforms.ToTensor()(data_loader_train)
#print(type(data_loader_train))

#images, labels = dataiter.next()
#print(type(images),type(labels))

transform1 = transforms.Compose(transforms.ToTensor())
#data_train = datasets.MNIST(root = '/Users/tianyiye/Documents/python_project/Dataset_test/2243_9243_bundle_archive/pytorch_download_dataset',
#                            transform=transform1,
#                            train = True,
#                            download = True)

#data_test = datasets.MNIST(root='/Users/tianyiye/Documents/python_project/Dataset_test/2243_9243_bundle_archive/pytorch_download_dataset',
#                          transform = transform1,
#                          train = False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                                shuffle = True)


#imshow(data_loader_train)
#imshow(data_name)
#plt.pause(0.5)