import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tf
#from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
#import math
#import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
#from skimage.io import imread
import os
#import numpy as np

from resnet import resnet34 as net
from net_aug import Set_prob, Set_var
from dataset import Labeledset, Unlabeledset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = net().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-6)

transform_train = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((256,256)),
        tf.ToTensor()
     ])

transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((256,256)),        
        tf.ToTensor(),
        tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ])

path1 = "/home/DATA/ymh/od/images/finetune"
path2 = "/home/DATA/ymh/od/labels/finetune"
save_path = "/home/DATA/ymh/od/outputs/finetune"

model.load_state_dict(torch.load(os.path.join(save_path, "epoch_050_loss_31.255897.pth")))

dataset_labeled = Labeledset(path1, path2, transform_train)
#dataset_unlabaled = Unlabeledset(path1, transform_train)
#dataset_val = Unlabeledset(path1, transform_valid)
#dataset_infer = Unlabeledset(path1, None)

labeled_batch = 4
#unlabeled_batch = 8 * 5
labeled_loader = DataLoader(dataset=dataset_labeled, batch_size=labeled_batch, shuffle=True)
#unlabeled_loader = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, shuffle=True)
#valid_loader = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False)

#lambda_1 = torch.linspace(0.5,1,100).to(device)
#sub_epoch = 20


Set_prob(model, None)
best = 20000
for epoch in range(50):
    print("="*100)
    
    running_loss = 0
    running_loss1 = 0
    running_loss2 = 0
    
    model.train()
    for x_L, y_L in labeled_loader:
    #for x_U in unlabeled_loader:
    #for _ in range(sub_epoch):
        #x_U = unlabeled_loader.__iter__().next()
        #x_U = x_U.float().to(device)      
        
        #x_L, y_L = labeled_loader.__iter__().next()
        x_L = x_L.float().to(device)
        y_L = y_L.float().to(device)
        
        optimizer.zero_grad()
        
        ## ---- Labeled set ---- ##
        #Set_prob(model, None)
        output_L = model(x_L)
        
        loss = criterion(output_L, y_L)
        
        ## ---- Unlabeled set ---- ##
        #Set_prob(model, 0.05)
        #output_U1 = model(x_U)

        #Set_prob(model, 0.05)
        #output_U2 = model(x_U)

        #loss_U = (criterion(output_U1, output_U2.clamp(0,256).detach()) + criterion(output_U2, output_U1.clamp(0,256).detach()))/2
        
        ## ---- Total ---- ##
        #loss = loss_L + lambda_1[epoch] * loss_U
        loss.backward()
        optimizer.step()
        
        #print(loss.item())
        running_loss += loss.item()
        #running_loss1 += loss_L.item()
        #running_loss2 += loss_U.item()
            
    running_loss /= len(labeled_loader)
    #running_loss1 /= sub_epoch # len(unlabeled_loader)
    #running_loss2 /= sub_epoch # len(unlabeled_loader)
    print("[Epoch:%d] [Loss:%f] [Labeled:%f] [Unlabeled:%f]" % ((epoch+1), running_loss, running_loss1, running_loss2))

    #if (epoch+1) % 2 == 0:
    #if running_loss < best:
    #    best = running_loss
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
