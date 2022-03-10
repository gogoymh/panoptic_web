import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
from skimage.io import imsave
from PIL import Image
import copy

from resnet import resnet34 as net_od
from Unet import Unet as net_seg
from net_aug import Set_prob, Set_var
from dataset2 import Labeledset, Unlabeledset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

od_model = net_od().to(device)
seg_model = net_seg().to(device)

transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((256,256)),        
        tf.ToTensor(),
        tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ])

path1 = "/home/DATA/ymh/od/images/004"
#path2 = "/DATA/ymh/panoptic/Labels/002"
#od_save_path = "/home/DATA/ymh/od/scripts"
od_save_path = "/home/DATA/ymh/od/outputs/finetune"

#od_result_path = "/DATA/ymh/panoptic/Outputs/002"

seg_save_path = "/home/DATA/ymh/od/Semantics/002"
output_path = "/home/DATA/ymh/od/Semantics/004/outputs"

dataset_infer = Unlabeledset(path1, None)

#od_model.load_state_dict(torch.load(os.path.join(od_save_path, "epoch_050_loss_31.255897.pth")))
od_model.load_state_dict(torch.load(os.path.join(od_save_path, "epoch_050_loss_38.132708.pth")))
od_model.eval()
Set_prob(od_model, None)

seg_model.load_state_dict(torch.load(os.path.join(seg_save_path, "epoch_061_loss_0.083556.pth")))
seg_model.eval()
Set_prob(seg_model, None)


colors = np.array([[255,227,213],
                   [7,11,53],
                   [255,255,255],
                   [67,41,0],
                   [159,129,114],
                   [212,110,120],
                   [79,41,44],
                   [255,255,0]])
mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

length = len(os.listdir(path1))
#length = len(os.listdir(od_result_path))
for index in range(length):    
#for index in range(10):
    img = dataset_infer.__getitem__(index)
            
    x_V = transform_valid(img).to(device)
            
    with torch.no_grad():
        output_V = od_model(x_V.unsqueeze(0)).squeeze()
                
    output_V = output_V.clamp(0,256)
    print(output_V, index)
            
    x1 = int(output_V[0]*(1920/256))
    y1 = int(output_V[1]*(1080/256))
    x2 = int(output_V[2]*(1920/256))
    y2 = int(output_V[3]*(1080/256))
    
    new_img = img[y1:y2,x1:x2,:]
    
    #imsave(os.path.join(output_path, "frame_%07d-1.png" % index), img)
    
    '''
    new_img = imread(os.path.join(od_result_path, "frame_%07d.png" % index))
    imsave(os.path.join(output_path, "frame_%07d-1.png" % index), new_img)
    '''
    W, H, _ = new_img.shape
    
    new_img = transform_valid(new_img).to(device)
    
    output = seg_model(new_img.unsqueeze(0))
    
    pred = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    pic = torch.zeros(3, 256, 256, dtype=torch.uint8)
    for i, k in enumerate(mapping):
        idx = pred == i
        pic[0][idx] = k[0]
        pic[1][idx] = k[1]
        pic[2][idx] = k[2]
        
    pic = pic.numpy().transpose(1,2,0)
    
    pic = resize(pic,
                 (W,H),
                 mode='edge',
                 anti_aliasing=False,
                 anti_aliasing_sigma=None,
                 preserve_range=True,
                 order=0)
                 
    pic = pic.astype('uint8')
    
    #imsave(os.path.join(output_path, "frame_%07d.png" % index), pic)
    
    '''
    pic = Image.open(os.path.join(output_path, "frame_%07d.png" % index))
    rgba = pic.convert("RGBA")
    datas = rgba.getdata()
  
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 0:  # finding yellow colour
            # replacing it with a transparent value
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
  
    rgba.putdata(newData)
    rgba.save(os.path.join(output_path, "frame_%07d-1.png" % index), "PNG")
    pic = imread(os.path.join(output_path, "frame_%07d-1.png" % index))
    
    img = Image.open(os.path.join(path1, "frame_%07d.png" % index))
    img = img.convert("RGBA")
    img = np.array(img)
    '''
    
    copy_img = copy.deepcopy(img)
    copy_img[y1:y2,x1:x2,:] = pic
    #imsave(os.path.join(output_path, "frame_%07d-1.png" % index), copy_img)
    
    img = Image.fromarray(img)
    img_rgba = img.convert("RGBA")
    copy_img = Image.fromarray(copy_img)
    copy_img_rgba = copy_img.convert("RGBA")
    
    
    orgs = img_rgba.getdata()
    copies = copy_img_rgba.getdata()
    
    newData = []
    for item1, item2 in zip(copies, orgs):
        
        if item1[0] == 255 and item1[1] == 255 and item1[2] == 0:
            newData.append((item2[0], item2[1], item2[2], 255))
        else:
            newData.append((item1[0], item1[1], item1[2], 255))
    
    copy_img_rgba.putdata(newData)
    copy_img_rgba.save(os.path.join(output_path, "frame_%07d-2.png" % index), "PNG")
    
    #imsave(os.path.join(output_path, "frame_%07d-2.png" % index), img)
    
    print(index, W, H)
    
    '''
    plt.imshow(img)
    ax = plt.gca()
    rect = Rectangle((x1,y1),(x2-x1),(y2-y1),linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.savefig(os.path.join(save_path, "frame_%07d.png" % (idx)))
    plt.show()
    plt.close()
    '''