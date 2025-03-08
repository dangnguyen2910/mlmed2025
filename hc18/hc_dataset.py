import pandas as pd 
import numpy as np 
import os 
import cv2
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import torch 
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2 as v2

class HCDataset(Dataset): 
    def __init__(self, path, transform = None, mask_transform = None): 
        self.data_path = path
        self.transform = transform
        self.mask_transform = mask_transform
    
    def __getitem__(self, index):
        all_img_list = os.listdir(self.data_path)
        img_list = [img for img in all_img_list if "Annotation" not in img]
        img_list.sort()
        img = cv2.imread(os.path.join(self.data_path, img_list[index]))

        if (self.transform): 
            img = self.transform(img)

        annot_list = [img for img in all_img_list if "Annotation" in img]
        annot_list.sort()
        mask = self.fill_head(os.path.join(self.data_path, annot_list[index]))
        
        if (self.mask_transform): 
            mask = self.mask_transform(mask)

        hc = self.get_head_circumference(img_list[index])
            
        gt = {}
        gt['mask'] = mask
        gt['hc'] = hc

        return img, gt

    def __len__(self): 
        return len(os.listdir(self.data_path))/2

    def fill_head(self, path): 
        """ 
        Fill ellipse in annotation image

        Parameter: 
        ---
        path (String): Path of annotation
        
        Output: 
        --- 
        numpy array of shape (w,h,1)
        """
        gt = cv2.imread(path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        threshold = threshold_otsu(gt)
        binary = gt <= threshold

        # Detect blob
        label_image = label(binary)

        mask = (label_image==2).astype(int)
        mask = mask[..., np.newaxis]
        return mask

    def get_head_circumference(self, path): 
        """ 
        Given image path, return its head circumference label
        
        Parameter: 
        ------------
        path(String): path of image

        Output: 
        -------
        Int: Head circumference of input image        
        """
        
        hc_df = pd.read_csv("data/hc18/training_set_pixel_size_and_HC.csv")
        row = hc_df[hc_df["filename"] == path]
        hc = row["head circumference (mm)"]
        return hc.item() 
        

if __name__ == "__main__": 
    dataset = Dataset()
    img, mask = Dataset[0]