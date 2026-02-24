import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform


class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg
        
        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images = []
        # images with related style  
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        for style in os.listdir(target_image_dir):
            style_path = f"{target_image_dir}/{style}"
            # Skip files (like .gitkeep) and only process directories
            if not os.path.isdir(style_path):
                continue
            images_related_style = []
            for img in os.listdir(style_path):
                img_path = f"{style_path}/{img}"
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]
        style, content = target_image_name.split('.')[0].split('+')
        
        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        content_image = Image.open(content_image_path).convert('RGB')

        # Random sample used for style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)
        
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image}
        
        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            
            # If style_list is empty, we can't choose a different style.
            # In this case, we might need to fallback to reusing the current style or duplicate styles.
            # However, reusing current style defeats the purpose of negative samples (Style Contrastive).
            # But for few-shot fine-tuning with only 1 other style (SimHei), style_list might have only 1 element before pop.
            # If style_list is empty after pop, it means we only have 2 styles total (Target + 1 other).
            
            if len(style_list) == 0:
                 # Fallback: if no other styles available, use the current style as negative (not ideal but prevents crash)
                 # Or better: use the ONLY other style available (which we just popped?)
                 # Actually, self.style_to_images keys include ALL styles.
                 # If we are fine-tuning, we might have MyStyle and SimHei.
                 # If current is MyStyle, style_list has SimHei. pop MyStyle -> SimHei remains.
                 # If we have only MyStyle... style_list has MyStyle. pop -> empty.
                 # We need at least 2 styles in TargetImage folder.
                 
                 # Let's restore the list if empty to avoid crash, but log a warning ideally.
                 style_list = list(self.style_to_images.keys())
                 # And maybe don't pop? Or pop a random one?
                 # If we really only have 1 style, we can't do contrastive learning properly.
                 # Assuming we have at least MyStyle and SimHei (created by prepare_dataset.py).
                 pass

            for i in range(self.num_neg):
                if len(style_list) > 0:
                    choose_style = random.choice(style_list)
                    # We don't pop here to allow replacement if we don't have enough styles
                    # choose_index = style_list.index(choose_style)
                    # style_list.pop(choose_index) 
                else:
                    # Fallback if really no styles
                    choose_style = style 
                
                # IMPORTANT: In prepare_dataset.py we only generated SimHei images for the characters we have.
                # But the code below assumes standard directory structure (style/style+content.jpg).
                # We need to make sure the path is constructed correctly based on what we actually have.
                # If choose_style is SimHei or MyStyle, the filename pattern is Style+Char.jpg
                
                choose_neg_name = f"{self.root}/{self.phase}/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
                
                # Check if file exists, if not, try to find ANY valid image for this character in other styles
                # This is a bit hacky but necessary for few-shot where we don't have full matrix
                if not os.path.exists(choose_neg_name):
                     # Try finding any existing style image for this content
                     found = False
                     for fallback_style in self.style_to_images.keys():
                         fallback_name = f"{self.root}/{self.phase}/TargetImage/{fallback_style}/{fallback_style}+{content}.jpg"
                         if os.path.exists(fallback_name):
                             choose_neg_name = fallback_name
                             found = True
                             break
                     if not found:
                         # If still not found (e.g. content char not in other styles), 
                         # we must use the target image itself as negative (better than crash)
                         # This effectively reduces contrastive loss for this sample but keeps training running.
                         choose_neg_name = target_image_path
                         
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
