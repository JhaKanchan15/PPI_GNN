
# note that this custom dataset is not prepared on the top of geometric Dataset(pytorch's inbuilt)
import os
import torch
import glob
import numpy as np 
import random
import math
from os import listdir
from os.path import isfile, join



processed_dir="../human_features/processed/"
npy_file = "../human_features/npy_file_new(human_dataset).npy"
npy_ar = np.load(npy_file)
print(npy_ar.shape)
from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import DataLoader as DataLoader_n

class LabelledDataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.load(npy_file)
      self.processed_dir = processed_dir
      self.protein_1 = self.npy_ar[:,2]
      self.protein_2 = self.npy_ar[:,5]
      self.label = self.npy_ar[:,6].astype(float)
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
      return(self.n_samples)

    def __getitem__(self, index):
      prot_1 = os.path.join(self.processed_dir, self.protein_1[index]+".pt")
      prot_2 = os.path.join(self.processed_dir, self.protein_2[index]+".pt")
      #print(f'Second prot is {prot_2}')
      prot_1 = torch.load(glob.glob(prot_1)[0])
      #print(f'Here lies {glob.glob(prot_2)}')
      prot_2 = torch.load(glob.glob(prot_2)[0])
      return prot_1, prot_2, torch.tensor(self.label[index])



dataset = LabelledDataset(npy_file = npy_file ,processed_dir= processed_dir)

final_pairs =  np.load(npy_file)
size = final_pairs.shape[0]
print("Size is : ")
print(size)
seed = 42
torch.manual_seed(seed)
#print(math.floor(0.8 * size))
#Make iterables using dataloader class  
trainset, testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size) ])
#print(trainset[0])
trainloader = DataLoader_n(dataset= trainset, batch_size= 4, num_workers = 0)
testloader = DataLoader_n(dataset= testset, batch_size= 4, num_workers = 0)
print("Length")
print(len(trainloader))
print(len(testloader))
