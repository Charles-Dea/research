import numpy as np
import pandas as pd
import pickle
from functools import reduce
import torch
from glob import glob
from itertools import cycle
from PIL import Image


class ImdbDataset:
    def __init__(self, config, split='train'):       
        data_paths = {'train': "datasets/aclImdb/train", 'eval': "datasets/aclImdb/test"}
        split_path = data_paths[split]
        neg_path = split_path + "/neg"
        pos_path = split_path + "/pos"
        neg_inputs = zip(glob(neg_path+"/*.txt"), cycle([0]))
        pos_inputs = zip(glob(pos_path+"/*.txt"), cycle([1]))
        self.data = np.random.permutation(list(neg_inputs) + list(pos_inputs))
        
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data[i]
        with open(data[0], 'r') as fo:
            source = fo.read()
        inputs = self.tokenizer(source, max_length=self.max_length)
        target = int(data[1])
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)


class ListOpsDataset:
    def __init__(self, config, split='train'):
        
        data_paths = {'train': "datasets/lra_release/listops-1000/basic_train.tsv",
                      'eval': "datasets/lra_release/listops-1000/basic_val.tsv"}
        self.data = pd.read_csv(data_paths[split], delimiter='\t')
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data.iloc[i]
        source = data.Source
        inputs = self.tokenizer(source, max_length=self.max_length) #return_tensors='pt', truncation=True, padding='max_length'
        target = data.Target
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)


class Cifar10Dataset:
    def __init__(self, config, split='train'):
        data_paths = {'train': [f"datasets/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)],
                      'eval': ["datasets/cifar-10-batches-py/test_batch"]
                     }
        print("loading cifar-10 data...")
        data_dicts = [Cifar10Dataset.unpickle(path) for path in data_paths[split]]
        print("assembling cifar-10 files..")
        self.data = reduce((lambda x, y: {b'data': np.concatenate([x[b'data'], y[b'data']], axis=0), 
                                         b'labels': np.concatenate([x[b'labels'], y[b'labels']], axis=0)}), 
                           data_dicts)
        # TODO CHECK: i think this is the right shape 
        # see: https://www.cs.toronto.edu/~kriz/cifar.html 
        #      section "Dataset layouts" discusses the memory layout of the array
        self.data[b'data'] = self.data[b'data'].reshape((-1, 3, 1024)) 
       
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
    
    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d
    
    def __getitem__(self, i):
        r, g, b = self.data[b'data'][i]
        # grayscale image (assume pixels in [0, 255])
        source = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(int)
        inputs = self.tokenizer(source, max_length=self.max_length)
        target = self.data[b'labels'][i]
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data[b'data'])


class Pathfinder32Dataset:
    def __init__(self,config,split='train'):
        mdf=200
        trnss=int(mdf*.85)
        self.data=[]
        self.lbls=[]
        dp='lra_release/lra_release/pathfinder32/curv_baseline/'
        mdp=dp+'metadata/'
        idp=dp+'imgs/'
        match split:
            case'train':
                for i in range(trnss):
                    self.lbls+=map(lambda x:x.split()[3]=='1',open(mdp+str(i)+'.npy'))
                    iidp=idp+str(i)+'/sample_'
                    for j in range(1000):
                        self.data.append(Image.open(iidp+str(j)+'.png').convert('L'))
            case'eval':
                for i in range(mdf-trnss):
                    self.lbls+=map(lambda x:x.split()[3]=='1',open(mdp+str(i)+'.npy'))
                    iidp=idp+str(i)+'/sample_'
                    for j in range(1000):
                        self.data.append(Image.open(iidp+str(j)+'.png').convert('L'))
        self.tokenizer=config.tokenizer
        self.max_length=config.max_length
    def __getitem__(self,i):
        return self.tokenizer(self.data[i],self.max_length),torch.LongTensor([self.lbls[i]])
    def __len__(self):
        return len(self.data)


class Pathfinder64Dataset:
    def __init__(self,config,split='train'):
        mdf=200
        trnss=int(mdf*.85)
        self.data=[]
        self.lbls=[]
        dp='lra_release/lra_release/pathfinder64/curv_baseline/'
        mdp=dp+'metadata/'
        idp=dp+'imgs/'
        match split:
            case'train':
                for i in range(trnss):
                    self.lbls+=map(lambda x:x.split()[3]=='1',open(mdp+str(i)+'.npy'))
                    iidp=idp+str(i)+'/sample_'
                    for j in range(1000):
                        self.data.append(Image.open(iidp+str(j)+'.png').convert('L'))
            case'eval':
                for i in range(mdf-trnss):
                    self.lbls+=map(lambda x:x.split()[3]=='1',open(mdp+str(i)+'.npy'))
                    iidp=idp+str(i)+'/sample_'
                    for j in range(1000):
                        self.data.append(Image.open(iidp+str(j)+'.png').convert('L'))
        self.tokenizer=config.tokenizer
        self.max_length=config.max_length
    def __getitem__(self,i):
        return self.tokenizer(self.data[i],self.max_length),torch.LongTensor([self.lbls[i]])
    def __len__(self):
        return len(self.data)


class Pathfinder128Dataset:
    def __init__(self,config,split='train'):
        mdf=200
        trnss=int(mdf*.85)
        self.data=[]
        self.lbls=[]
        dp='lra_release/lra_release/pathfinder128/curv_baseline/'
        mdp=dp+'metadata/'
        idp=dp+'imgs/'
        match split:
            case'train':
                for i in range(trnss):
                    self.lbls+=map(lambda x:x.split()[3]=='1',open(mdp+str(i)+'.npy'))
                    iidp=idp+str(i)+'/sample_'
                    for j in range(1000):
                        self.data.append(Image.open(iidp+str(j)+'.png').convert('L'))
            case'eval':
                for i in range(mdf-trnss):
                    self.lbls+=map(lambda x:x.split()[3]=='1',open(mdp+str(i)+'.npy'))
                    iidp=idp+str(i)+'/sample_'
                    for j in range(1000):
                        self.data.append(Image.open(iidp+str(j)+'.png').convert('L'))
        self.tokenizer=config.tokenizer
        self.max_length=config.max_length
    def __getitem__(self,i):
        return self.tokenizer(self.data[i],self.max_length),torch.LongTensor([self.lbls[i]])
    def __len__(self):
        return len(self.data)


class Pathfinder256Dataset:
    """
    def __init__(self,config,split='train'):
        mdf=200
        trnss=int(mdf*.85)
        self.data=[]
        self.lbls=[]
        dp='lra_release/lra_release/pathfinder256/curv_baseline/'
        mdp=dp+'metadata/'
        idp=dp+'imgs/'
        match split:
            case'train':
                for i in range(trnss):
                    self.lbls+=map(lambda x:x.split()[3]=='1',open(mdp+str(i)+'.npy'))
                    iidp=idp+str(i)+'/sample_'
                    for j in range(1000):
                        self.data.append(Image.open(iidp+str(j)+'.png').convert('L'))
            case'eval':
                for i in range(mdf-trnss):
                    self.lbls+=map(lambda x:x.split()[3]=='1',open(mdp+str(i)+'.npy'))
                    iidp=idp+str(i)+'/sample_'
                    for j in range(1000):
                        self.data.append(Image.open(iidp+str(j)+'.png').convert('L'))
        self.tokenizer=config.tokenizer
        self.max_length=config.max_length
    """
    def __init__(self,config,split='train'):
        self.lbls=[]
        for i in range(200):
            self.lbls+=map(lambda x:x.split()[3]=='1',open('lra_release/lra_release/pathfinder256/curv_baseline/metadata/'+str(i)+'.npy'))
        self.tokenizer=config.tokenizer
        self.max_length=config.max_length
        self.split=split
    def __getitem__(self,i):
        fn='lra_release/lra_release/pathfinder256/curv_baseline/imgs/'+str(i//1000+200*(.85 if self.split=='eval'else 0))+'/sample_'+str(i%1000)+'.png'
        return self.tokenizer(Image.open(fn).convert('L'),self.max_length),torch.LongTensor([self.lbls[i]])
        #return self.tokenizer(self.data[i],self.max_length),torch.LongTensor([self.lbls[i]])
    def __len__(self):
        return len(self.lbls)
