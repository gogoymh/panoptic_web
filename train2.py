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
import os
import numpy as np

from Unet import Unet as net
from net_aug import Set_prob, Set_var
from dataset2 import Labeledset, Unlabeledset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)

path1 = "/DATA/ymh/panoptic/Semantics/002/images"
path2 = "/DATA/ymh/panoptic/Semantics/002/numpys"

path3 = "/DATA/ymh/panoptic/Outputs/002"

#seg_save_path = "/home/compu/ymh/od/outputs/002/seg"

save_path = "/DATA/ymh/panoptic/Semantics/002"

dataset_labeled = Labeledset(path1, path2)
dataset_unlabaled = Unlabeledset(path3, True)

labeled_batch = 4
unlabeled_batch = 4
labeled_loader = DataLoader(dataset=dataset_labeled, batch_size=labeled_batch, shuffle=True)
unlabeled_loader = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, shuffle=True)

lambda_1 = torch.linspace(1e-6,0.5,300).to(device)
sub_epoch = 30

#model.load_state_dict(torch.load(os.path.join(seg_save_path, "epoch_006_loss_0.013961.pth")))

best = 200
for epoch in range(300):
    print("="*100)
    
    running_loss = 0
    running_loss1 = 0
    running_loss2 = 0
    
    model.train()
    #for x_U in unlabeled_loader:
    for _ in range(sub_epoch):
        x_U = unlabeled_loader.__iter__().next()
        x_U = x_U.float().to(device)      
        
        x_L, y_L = labeled_loader.__iter__().next()
        x_L = x_L.float().to(device)
        y_L = y_L.long().to(device)
        
        optimizer.zero_grad()
        
        ## ---- Labeled set ---- ##
        Set_prob(model, None)
        output_L = model(x_L)
        
        loss_L = criterion(output_L, y_L)
        '''
        Set_prob(model, 0.05)
        output_L1 = model(x_L)

        Set_prob(model, 0.05)
        output_L2 = model(x_L)
        
        loss_U = (criterion(output_L1, output_L2.argmax(dim=1).detach()) + criterion(output_L2, output_L1.argmax(dim=1).detach()))/2
        '''
        
        ## ---- Unlabeled set ---- ##
        Set_prob(model, 0.05)
        output_U1 = model(x_U)

        Set_prob(model, 0.05)
        output_U2 = model(x_U)

        loss_U = (criterion(output_U1, output_U2.argmax(dim=1).detach()) + criterion(output_U2, output_U1.argmax(dim=1).detach()))/2
        
        ## ---- Total ---- ##
        loss = loss_L + lambda_1[epoch] * loss_U
        loss.backward()
        optimizer.step()
        
        #print(loss.item())
        running_loss += loss.item()
        running_loss1 += loss_L.item()
        running_loss2 += loss_U.item()
            
    running_loss /= sub_epoch# len(unlabeled_loader) # 
    running_loss1 /= sub_epoch# len(unlabeled_loader)
    running_loss2 /= sub_epoch# len(unlabeled_loader)
    print("[Epoch:%d] [Loss:%f] [Labeled:%f] [Unlabeled:%f]" % ((epoch+1), running_loss, running_loss1, running_loss2))

    if (epoch+1) % 1 == 0:
    #if running_loss < best:
        #best = running_loss
        model.eval()
        Set_prob(model, None)
        torch.save(model.state_dict(), os.path.join(save_path, "epoch_%03d_loss_%f.pth" % (epoch+1, running_loss)))
        ## ---- Sparse1 ---- ##
        '''
        model.eval()
        Set_prob(model, None)

        #for idx, x_V in enumerate(valid_loader):
        for idx in range(20):
            #x_V = valid_loader.__iter__().next()
            #x_V = x_V.float().to(device)
            
            img = dataset_infer.__getitem__(idx)
            
            x_V = transform_valid(img).to(device)
            
            with torch.no_grad():
                output_V = model(x_V.unsqueeze(0)).squeeze()
                
                output_V = output_V.clamp(0,256)
                #print(output_V)            
            
            x1 = int(output_V[0]*(1920/256))
            y1 = int(output_V[1]*(1080/256))
            x2 = int(output_V[2]*(1920/256))
            y2 = int(output_V[3]*(1080/256))
            
            plt.imshow(img)
            ax = plt.gca()
            rect = Rectangle((x1,y1),(x2-x1),(y2-y1),linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            plt.savefig(os.path.join(save_path, "epoch_%03d_frame_%07d.png" % (epoch+1, idx)))
            plt.show()
            plt.close()
        '''
