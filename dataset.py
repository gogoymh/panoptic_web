import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms
import torch.nn.functional as F
import cv2

def RandomScale(img, bboxes):
        img_shape = img.shape
        
        random_factor = np.random.uniform(-0.2,0.2,2)
        
        scale_x = random_factor[0]
        scale_y = random_factor[1]
        
        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y
        
        img = cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
        
        bboxes[0] *= resize_scale_x
        bboxes[1] *= resize_scale_y
        bboxes[2] *= resize_scale_x
        bboxes[3] *= resize_scale_y
        
        canvas = np.zeros(img_shape, dtype = np.uint8)
        
        y_lim = int(min(resize_scale_y,1)*img_shape[0])
        x_lim = int(min(resize_scale_x,1)*img_shape[1])
        
        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
        
        img = canvas
    
    
        return img, bboxes 

def translate(img, bboxes):        
        img_shape = img.shape
        
        random_factor = np.random.uniform(-0.2,0.2,2)
        
        translate_factor_x = random_factor[0]
        translate_factor_y = random_factor[1]
            
        canvas = np.zeros(img_shape).astype(np.uint8)

        corner_x = int(translate_factor_x*img.shape[1])
        corner_y = int(translate_factor_y*img.shape[0])
        
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes[0] += corner_x
        bboxes[1] += corner_y
        bboxes[2] += corner_x
        bboxes[3] += corner_y

        return img, bboxes

def get_index(name):
    return int(name.split("_")[1].split(".")[0])

class Labeledset(Dataset):
    def __init__(self, path1, path2, transform, width=1920, height=1080):
        super().__init__()
        
        self.path1 = path1
        self.path2 = path2
        self.resize = [256/width, 256/height]
        
        self.transform = transform
        self.normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        
        name = os.listdir(path2)
        self.name = list(map(get_index, name))
        
        self.len = len(self.name)
        
    def __getitem__(self, index):
        img = imread(os.path.join(self.path1, "frame_%07d.png" % self.name[index]))
        
        file = open(os.path.join(self.path2, "frame_%07d.txt" % self.name[index]), "r")
        strings = file.readlines()
        x1, y1, x2, y2 = strings[1].split(" ")
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2[:-1])
        
        label = np.array([x1, y1, x2, y2])
        
        if self.transform is not None:
            img, label = translate(img, label)
            img, label = RandomScale(img, label)
            
            #seed = np.random.randint(2147483647)
            #torch.manual_seed(seed)
            #random.seed(seed)
            img = self.transform(img)
            img = self.normalize(img)
            
            label[0] *= self.resize[0]
            label[1] *= self.resize[1]
            label[2] *= self.resize[0]
            label[3] *= self.resize[1]
            
            label = torch.from_numpy(label).clamp(0,256)
        
        return img, label
    
    def __len__(self):
        return self.len

class Unlabeledset(Dataset):
    def __init__(self, path1, transform):
        super().__init__()
        
        self.path1 = path1
        
        self.transform = transform
        self.normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        
        name = os.listdir(path1)
        self.name = list(map(get_index, name))
        
        self.len = len(self.name)
        
    def __getitem__(self, index):
        img = imread(os.path.join(self.path1, "frame_%07d.png" % self.name[index]))
        
        seed = np.random.randint(2147483647)
        if self.transform is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            img = self.transform(img)
            img = self.normalize(img)
        
        return img
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from torchvision import transforms as tf
    
    path1 = "C://BBox-Label-Tool//Images//002//"
    path2 = "C://BBox-Label-Tool//Labels//002//"
    
    transform = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((256,256)),
        tf.ToTensor(),
     ])
    
    a = Labeledset(path1, path2, transform)
    img, label = a.__getitem__(0)
    
    print(label)
    
    img = img.numpy().transpose(1,2,0)
    plt.imshow(img)
    
    x1 = int(label[0])#*(1920/256))
    y1 = int(label[1])#*(1080/256))
    x2 = int(label[2])#*(1920/256))
    y2 = int(label[3])#*(1080/256))

    
    ax = plt.gca()
    rect = Rectangle((x1,y1),(x2-x1),(y2-y1),linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()
    plt.close()