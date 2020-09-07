import torch
import torchvision
from torchvision import datasets,transforms
import torch.utils.data as data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE = 128
NUM_EPOCHS = 10
# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

data_train = datasets.MNIST(root = 'D:\TianyiYe\Dataset_test\datasets',
                          transform=transform,
                           train = True,
                           download = True)

data_test = datasets.MNIST(root = 'D:\TianyiYe\Dataset_test\datasets',
                         transform = transform,
                        train = False)

print(type(data_train))
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size = BATCH_SIZE,
                                                shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size = BATCH_SIZE,
                                                shuffle = False)
print(type(data_loader_train))
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(data_loader_train)
images, labels = dataiter.next()#how images becomes a tensor?
print(type(labels))
imshow(torchvision.utils.make_grid(images))
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))