import os
import torch
import numpy as np
import random

from torch.utils.data import Dataset
from PIL import Image
from utilities import resize

class mydata(Dataset):
    def __init__(self, LR_path, GT_path, in_memory=True, transform=None):
        
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.transform = transform
        self.in_memory = in_memory
        
        self.LR_img = sorted(os.listdir(LR_path))
        self.GT_img = sorted(os.listdir(GT_path))

        # scale GT to 64x64 if larger
        test_dim = Image.open(os.path.join(self.GT_path, self.GT_img[0])).convert("RGB")
        if test_dim.size[0] > 64:
            resize.rewrite(self.GT_path, self.GT_img)

        # get array of LR and HR images
        if self.in_memory:
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr)).convert("RGB")).astype(np.uint8) for lr in self.LR_img]
            self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, gt)).convert("RGB")).astype(np.uint8) for gt in self.GT_img]


    def __len__(self):    

        return len(self.LR_img)
    

    def __getitem__(self, i):

        img_item = {}

        if self.in_memory:    
            GT = self.GT_img[i].astype(np.float32)
            LR = self.LR_img[i].astype(np.float32)
        else:
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i])).convert("RGB"))
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])).convert("RGB"))

        img_item['GT'] = (GT / 127.5) - 1.0
        img_item['LR'] = (LR / 127.5) - 1.0
                
        if self.transform is not None:
            img_item = self.transform(img_item)
        
        # np/cv2 read images as h, w, c. Instead I want c, h, w
        # FYI: PIL reads as w, h, (c)
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        
        return img_item
    
    
class testOnly_data(Dataset):
    def __init__(self, LR_path, in_memory=True, transform = None):
        
        self.in_memory = in_memory
        self.LR_path = LR_path
        self.LR_img = sorted(os.listdir(LR_path))
        
        if self.in_memory:
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr))) for lr in self.LR_img]
        
    def __len__(self):
        
        return len(self.LR_img)
        
    def __getitem__(self, i):
        
        img_item = {}
        
        if self.in_memory:
            LR = self.LR_img[i]
        else:
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])))
            
        img_item['LR'] = (LR / 127.5) - 1.0                
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        
        return img_item


class crop(object):
    """ crop

    crop will be part of transforms.Compose
    Transforms is passed a dictionary (of keys = {'HR' and 'LR'}) as input.
    In this case the image dictionary is called 'sample'.

    To access the images:
        - dict['LR']
        - dict['HR']
    """

    def __init__(self, scale, patch_size):
        
        self.scale = scale
        self.patch_size = patch_size
    
    def __call__(self, sample):
        """ built-in method

        the instances behave like functions and can be called like a function
        
        e.g.
        c = crop(scale, patch)
        c(sample)   # calls a function directly from the object instance itself

        'sample' is the dictionary of images
        """

        LR_img, GT_img = sample['LR'], sample['GT']
        ih, iw = LR_img.shape[:2]       # after transpose(2,0,1) = h, w
        
        # LR patch
        ix = random.randrange(0, abs(iw - self.patch_size +1))
        iy = random.randrange(0, abs(ih - self.patch_size +1))
        
        # HR patch
        tx = ix * self.scale
        ty = iy * self.scale

        # match the correct pixels patch from LR to HR
        # potrei croppare l'intera immagine così non scassa il cazzo       
        # LR_patch = LR_img[iy : iy + self.patch_size, ix : ix + self.patch_size]
        # GT_patch = GT_img[ty : ty + (self.scale * self.patch_size), tx : tx + (self.scale * self.patch_size)]
        LR_patch = LR_img[0 : self.patch_size, 0 : self.patch_size]
        GT_patch = GT_img[0 : (self.scale * self.patch_size), 0 : (self.scale * self.patch_size)]

        return {'LR' : LR_patch, 'GT' : GT_patch}

class augmentation(object):
    
    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']
        
        hor_flip = random.randrange(0,2)
        ver_flip = random.randrange(0,2)
        rot = random.randrange(0,2)
    
        if hor_flip:
            temp_LR = np.fliplr(LR_img)
            LR_img = temp_LR.copy()
            temp_GT = np.fliplr(GT_img)
            GT_img = temp_GT.copy()
            
            del temp_LR, temp_GT
        
        if ver_flip:
            temp_LR = np.flipud(LR_img)
            LR_img = temp_LR.copy()
            temp_GT = np.flipud(GT_img)
            GT_img = temp_GT.copy()
            
            del temp_LR, temp_GT
            
        if rot:
            LR_img = LR_img.transpose(1, 0, 2)
            GT_img = GT_img.transpose(1, 0, 2)
        
        
        return {'LR' : LR_img, 'GT' : GT_img}
        

