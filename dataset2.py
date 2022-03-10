import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms
import torch.nn.functional as F
import cv2

def get_index(name):
    return int(name.split("_")[1].split(".")[0])

class Labeledset(Dataset):
    def __init__(self, path1, path2):
        super().__init__()
        
        self.img_path = path1
        self.segmantic_path = path2
        
        self.files = os.listdir(path2)
        self.files.sort()
        #print(self.files)
        
        self.basic_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        self.spatial_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256))
                ])  
        
        self.pil = transforms.ToPILImage()
        
        self.augmentation = transforms.Compose([
                transforms.RandomResizedCrop((256,256), scale=(0.5, 1.5)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomAffine(0, translate=(0.2,0.2))                
                transforms.RandomAffine(0, shear=[-10, 10, -10, 10])
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.len = len(self.files)
        print("Dataset Length is %d" % self.len)
        
    def __getitem__(self, index):
        name = self.files[index]
        
        mask = np.load(os.path.join(self.segmantic_path, name))
        img = imread(os.path.join(self.img_path, "frame_%07d.png" % int(name.split("_")[1].split(".")[0])))
        
        img = self.spatial_transform(img[:,:,:3])
        mask = mask.astype('uint8')
        mask = torch.from_numpy(mask)
        mask = self.pil(mask)
        
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)
        img = self.augmentation(img)
        torch.manual_seed(seed)
        random.seed(seed)
        mask = self.augmentation(mask)
        
        img = self.normalize(img)
        mask = self.basic_transform(mask) * 255
        
        return img, mask.squeeze().long()
    
    def __len__(self):
        return self.len

class Unlabeledset(Dataset):
    def __init__(self, path1, ifnone=None):
        super().__init__()
        
        self.path1 = path1
        
        if ifnone is not None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.RandomResizedCrop((256,256), scale=(0.5, 1.5)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomAffine(0, translate=(0.2,0.2))                
                transforms.RandomAffine(0, shear=[-10, 10, -10, 10]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = None
        
        
        name = os.listdir(path1)
        self.name = sorted(list(map(get_index, name)))
        print(self.name[:10])
        
        self.len = len(self.name)
        
    def __getitem__(self, index):
        img = imread(os.path.join(self.path1, "frame_%07d.png" % self.name[index]))
        
        seed = np.random.randint(2147483647)
        if self.transform is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            img = self.transform(img)
        
        return img
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    path1 = "C://BBox-Label-Tool//Semantics//002//images//"
    path2 = "C://BBox-Label-Tool//Semantics//002//numpys//"
    
    #path1 = "/home/compu/ymh/od/Semantics/002/images"
    #path2 = "/home/compu/ymh/od/Semantics/002/numpys"
    
    a = Labeledset(path1, path2)
    b, c = a.__getitem__(0)
        
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
       
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    plt.savefig("img.png")
    plt.show()
    plt.close()
    
    colors = np.array([[255,227,213],
                   [7,11,53],
                   [255,255,255],
                   [67,41,0],
                   [159,129,114],
                   [212,110,120],
                   [79,41,44],
                   [255,255,0]])
    
    mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
    
    pic = torch.zeros(3, 256, 256, dtype=torch.uint8)
    for i, k in enumerate(mapping):
        idx = c == i
        pic[0][idx] = k[0]
        pic[1][idx] = k[1]
        pic[2][idx] = k[2]
        
    pic = pic.numpy().transpose(1,2,0)
    plt.imshow(pic)
    plt.savefig("label.png")
    plt.show()
    plt.close()
    
    
    path3 = "C://BBox-Label-Tool//Images//002//"
    #path3 = "/home/compu/ymh/od/outputs/002/outputs"
    a = Unlabeledset(path3)
    
    b = a.__getitem__(0)
        
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
       
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    plt.savefig("img2.png")
    plt.show()
    plt.close()
    