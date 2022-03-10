import torch
from torchvision import transforms as tf
import os
from skimage.io import imsave

from resnet import resnet34 as net
from net_aug import Set_prob
from dataset import Unlabeledset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = net().to(device)


transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((256,256)),        
        tf.ToTensor(),
        tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ])

path1 = "/DATA/ymh/panoptic/Images/002"
save_path = "/DATA/ymh/panoptic/scripts"
img_path = "/DATA/ymh/panoptic/Outputs/002"

dataset_infer = Unlabeledset(path1, None)

model.load_state_dict(torch.load(os.path.join(save_path, "epoch_050_loss_31.255897.pth")))
model.eval()
Set_prob(model, None)

#for idx, x_V in enumerate(valid_loader):
    
length = len(os.listdir(path1))
for idx in range(550,length):
    img = dataset_infer.__getitem__(idx)
            
    x_V = transform_valid(img).to(device)
            
    with torch.no_grad():
        output_V = model(x_V.unsqueeze(0)).squeeze()
                
    output_V = output_V.clamp(0,256)
    print(output_V, idx)
            
    x1 = int(output_V[0]*(1920/256))
    y1 = int(output_V[1]*(1080/256))
    x2 = int(output_V[2]*(1920/256))
    y2 = int(output_V[3]*(1080/256))
    
    imsave(os.path.join(img_path, "frame_%07d.png" % idx), img[y1:y2,x1:x2,:])
    
    '''
    plt.imshow(img)
    ax = plt.gca()
    rect = Rectangle((x1,y1),(x2-x1),(y2-y1),linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.savefig(os.path.join(save_path, "frame_%07d.png" % (idx)))
    plt.show()
    plt.close()
    '''