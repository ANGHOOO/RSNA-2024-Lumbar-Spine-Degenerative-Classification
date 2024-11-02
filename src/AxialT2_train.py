import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import gc

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastai.vision.all import *
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1337
FOLDS = [1,2,3,4,5]
PATH = 'C:/Users/Angel/kaggle/'# Main path
TRAIN_PATH = 'C:/Users/Angel/kaggle/train/'# Training images folder
ENCODER_NAME = "resnet18"
ANGLE = 180#30
S2 = 64
PATCH_H = 512
PATCH_W = 512
BS = 16
LR = 2.5e-4
EPOCHS = 1
TH = .5

S2 = torch.as_tensor(S2)
A = -1/(2*S2).to(device)

train = pd.read_csv(PATH + 'train_split.csv')
train.tail()

df_coor = pd.read_csv(PATH + 'train_label_coordinates.csv')
df_coor.tail()

S = df_coor[
    df_coor['condition'].isin([
        'Left Subarticular Stenosis',
        'Right Subarticular Stenosis'
    ])
].sort_values([
    'study_id',
    'series_id',
    'level',
    'condition'
]).reset_index(drop=True)
S.tail()

centers = {}
for i in range(len(S)):
    row = S.iloc[i]
    centers[row['study_id']]={}
for i in range(len(S)):
    row = S.iloc[i]
    centers[row['study_id']][row['series_id']]={}
for i in range(len(S)):
    row = S.iloc[i]
    centers[row['study_id']][row['series_id']][row['instance_number']]={'L':[], 'R':[], 'level':[]}
for i in range(len(S)):
    row = S.iloc[i]
    centers[row['study_id']][row['series_id']][row['instance_number']][row['condition'][0]].append([row['x'],row['y']])

S = S[[
    'study_id',
    'series_id',
    'instance_number'
]].groupby([
    'study_id',
    'series_id',
    'instance_number'
]).count().reset_index()

coordinates = np.zeros((len(S),4))
coordinates[:] = np.nan
for i in range(len(S)):
    row = S.iloc[i]
    for side in ['L','R']:
            if len(centers[row['study_id']][row['series_id']][row['instance_number']][side]) > 0:
                c = np.array(centers[row['study_id']][row['series_id']][row['instance_number']][side]).mean(0)
                coordinates[
                    i,
                    {'L':0, 'R':2}[side]:{'L':0, 'R':2}[side]+2
                ] = c

coor = [
    'x_L',
    'y_L',
    'x_R',
    'y_R'    
]

S[coor] = coordinates
S.tail()

S[S[coor].isna().sum(1) > 0].reset_index(drop=True).tail()

centroids = S[S[coor].isna().sum(1) == 0].reset_index(drop=True)
centroids.tail()

for (study_id,series_id),df in tqdm(centroids.groupby(['study_id','series_id'])):
    sample = TRAIN_PATH + str(study_id) + '/' + str(series_id)
    instance_numbers = [int(x.replace('\\','/').split('/')[-1].replace('.dcm','')) for x in glob.glob(sample+'/*.dcm')]
    instance_numbers.sort()
    instance_numbers = np.array(instance_numbers)
    D = len(instance_numbers)
    for i in range(len(df)):
        row = df[i:i+1]
        instance_number = row['instance_number'].values
        instance_number_index = int(np.arange(D)[instance_numbers == instance_number])
        if instance_number_index > 0:
            new = row.copy()
            new['instance_number'] = instance_numbers[instance_number_index - 1]
            centroids = pd.concat([
                centroids,
                new
            ])
        if instance_number_index < D - 1:
            new = row.copy()
            new['instance_number'] = instance_numbers[instance_number_index + 1]
            centroids = pd.concat([
                centroids,
                new
            ])

S = pd.concat([S[S[coor].isna().sum(1) > 0],centroids]).groupby(['study_id','series_id','instance_number']).mean().reset_index()
S.tail()

S[S[coor].isna().sum(1) > 0].reset_index(drop=True).tail()

S['flip'] = False
fS = S.copy()
fS['flip'] = True
S = pd.concat([S,fS]).reset_index(drop=True)

S = S.merge(train[['study_id','fold']], left_on='study_id', right_on='study_id')
S.tail()

df_meta_f = pd.read_csv(PATH + 'train_series_descriptions.csv')
df_meta_f.tail()

S = S.merge(df_meta_f[['series_id','series_description']], left_on='series_id', right_on='series_id')
S.tail()

S.groupby('series_description').count()

S.groupby('fold').count()

def augment_image_and_centers(image,centers,center=(PATCH_H/2,PATCH_W/2)):
    # Randomly rotate the image.
    angle = torch.as_tensor(random.uniform(-ANGLE, ANGLE))
    image = torchvision.transforms.functional.rotate(
        image,angle.item(),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        center=center
    )
    # https://discuss.pytorch.org/t/rotation-matrix/128260
    angle = -angle*math.pi/180
    s = torch.sin(angle)
    c = torch.cos(angle)
    rot = torch.stack([
        torch.stack([c, s]),
        torch.stack([-s, c])
    ])
    center = torch.as_tensor(center).float()
    centers = ((centers.cpu() - center) @ rot) + center

    return image,centers

torch_resize = torchvision.transforms.Resize((PATCH_H,PATCH_W),antialias=True)

x_map = torch.stack([torch.arange(PATCH_W)]*PATCH_H).float()
y_map = torch.stack([torch.arange(PATCH_H)]*PATCH_W).float()
idx_map = torch.stack([x_map,y_map.T]).view(1,2,PATCH_H,PATCH_W).to(device)

class Axial_T2_axial_side_Dataset(Dataset):
    def __init__(self, df, VALID=False, alpha=0):
        self.data = df
        self.VALID = VALID
        self.alpha = alpha

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        centers = torch.as_tensor([x for x in row[coor]]).view(2,2).float().to(device)
        
        sample = TRAIN_PATH + str(row['study_id']) + '/'+ str(row['series_id']) + '/'+ str(row['instance_number']) + '.dcm'
        
        image = pydicom.dcmread(sample).pixel_array
        H,W = image.shape

        if H > W:
            d = W
            if not self.VALID:
                h = int((H - d)*(.5 + self.alpha*(.5 - np.random.rand())))
            else:
                h = (H - d)//2
            image = image[h:h+d]
            centers[:,1] -= h
            H = W
        elif H < W:
            d = H
            if not self.VALID:
                w = int((W - d)*(.5 + self.alpha*(.5 - np.random.rand())))
            else:
                w = (W - d)//2
            image = image[:,w:w+d]
            centers[:,0] -= w
            W = H
        image = torch_resize(torch.as_tensor((image/np.max(image)).astype(np.float32)).unsqueeze(0))
        image = image.float().to(device)
        
        centers[:,0] = centers[:,0]*PATCH_W/W
        centers[:,1] = centers[:,1]*PATCH_H/H

        if not self.VALID: image,centers = augment_image_and_centers(image,centers,centers.nanmean(0).tolist())

        if row['flip']:
            image = image.flip(2)
            centers = centers.flip(0)
            centers[:,0] = PATCH_W - centers[:,0] - 1

        return image,centers

tds = Axial_T2_axial_side_Dataset(S)

for k in range(10):
    image,centers = tds.__getitem__(np.random.randint(len(tds)))
    centers = centers[centers.isnan().sum(1) == 0]
#   Ideal heatmaps
    mask = idx_map - centers.view(len(centers),2,1,1).to(device)
    mask = (mask*mask).sum(1)
    mask = torch.exp(A*mask)
    mask = mask.sum(0)
    plt.imshow(image.cpu()[0] + .5*(mask.cpu() > TH))
    plt.show()

vds = Axial_T2_axial_side_Dataset(S,VALID=True)

for k in range(10):
    image,centers = vds.__getitem__(np.random.randint(len(vds)))
    centers = centers[centers.isnan().sum(1) == 0]
#   Ideal heatmaps
    mask = idx_map - centers.view(len(centers),2,1,1).to(device)
    mask = (mask*mask).sum(1)
    mask = torch.exp(A*mask)
    mask = mask.sum(0)
    plt.imshow(image.cpu()[0] + .5*(mask.cpu() > TH))
    plt.show()

del tds,vds
gc.collect()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class myUNet(nn.Module):
    def __init__(
        self,
        classes
        ):
        super(myUNet, self).__init__()

        self.classes = classes
        self.UNet = smp.Unet(
            encoder_name=ENCODER_NAME,
            classes=classes,
            in_channels=1
        ).to(device)

    def forward(self,X):
        H,W = X.shape[-2:]
        x = self.UNet(X.view(-1,1,H,W)).view(-1,H*W)
#       MinMaxScaling along the class plane to generate a heatmap
        min_values = x.min(-1)[0].view(-1,1)
        max_values = x.max(-1)[0].view(-1,1)
        d = (max_values - min_values)
        d[d == 0] = 1
        x = (x - min_values)/d
        
        return x.view(-1,self.classes,H,W)

class myLoss(nn.Module):
    def __init__(
            self,
            alpha=.5,
            smooth = 1e-6
        ):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def clone(self):
        return myLoss(self.alpha)

    def forward(
            self,
            heatmaps,# Predictions
            centers # Targets
        ):
        H,W = heatmaps.shape[-2:]
        heatmaps = heatmaps.view(-1,H*W)
        centers = centers.view(-1,2)
        m = centers.isnan().sum(1) == 0
        heatmaps = heatmaps[m]
        centers = centers[m]
#       Ideal heatmaps
        mask = idx_map - centers.view(len(centers),2,1,1).to(device)
        mask = (mask*mask).sum(1)
        mask = torch.exp(A*mask)
        mask = mask.view(-1,H*W)
#       Distance
        D = 1 - ((mask*heatmaps).sum(-1))**2/((mask*mask).sum(-1)*(heatmaps*heatmaps).sum(-1)+self.smooth)
        
        return D.mean()

# CosineAnnealingAlpha
def nt(nmin,nmax,tcur,tmax):
    return (nmax - .5*(nmax-nmin)*(1+np.cos(tcur*np.pi/tmax))).astype(np.float32)

# callback to update alpha during training
def cb(self):
    alpha = torch.as_tensor(nt(.25,1,learn.train_iter,EPOCHS*n_iter))
    learn.dls.train_ds.alpha = alpha
alpha_cb = Callback(before_batch=cb)

for f in FOLDS:
    seed_everything(SEED)
    model = myUNet(2)
    
    tdf = S[S['fold'] != f]
    vdf = S[S['fold'] == f]

    tds = Axial_T2_axial_side_Dataset(tdf)
    vds = Axial_T2_axial_side_Dataset(vdf,VALID=True)
    
    tdl = torch.utils.data.DataLoader(tds, batch_size=BS, shuffle=True, drop_last=True)
    vdl = torch.utils.data.DataLoader(vds, batch_size=BS, shuffle=False)

    dls = DataLoaders(tdl,vdl)

    n_iter = len(tds)//BS

    learn = Learner(
        dls,
        model,
        lr=LR,
        loss_func=myLoss(alpha=0.5),
        cbs=[
            ShowGraphCallback(),
            alpha_cb
        ]
    )
    learn.fit_one_cycle(EPOCHS)
    torch.save(model,'Axial_T2_axial_side_segmentation_'+str(f))
    del tdl,vdl,dls,model,learn
    gc.collect()

f = 1
model = torch.load('Axial_T2_axial_side_segmentation_'+str(f))
vdf = S[S['fold'] == f]
vds = Axial_T2_axial_side_Dataset(vdf,VALID=True)

for k in range(20):
    i = np.random.randint(len(vds))
    print(i)
    image,centers = vds.__getitem__(np.random.randint(len(vds)))
    centers = centers[centers.isnan().sum(1) == 0]
#   Ideal heatmaps
    mask = idx_map - centers.view(len(centers),2,1,1).to(device)
    mask = (mask*mask).sum(1)
    mask = torch.exp(A*mask)
    mask = mask.sum(0)
    fig, axes = plt.subplots(1, 2, figsize=(10,10))
    axes[0].imshow(image.cpu()[0] + .5*(model(image.unsqueeze(0))[0].detach().cpu() > TH).sum(0))
    axes[1].imshow(image.cpu()[0] + .5*(mask.cpu() > TH))
    plt.show()

del model,vdf,vds
gc.collect()