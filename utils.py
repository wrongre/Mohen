import os
import cv2
import yaml
import copy
import pygame
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

import torch
import torchvision.transforms as transforms

def save_args_to_yaml(args, output_file):
    # Convert args namespace to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def save_single_image(save_dir, image):

    save_path = f"{save_dir}/out_single.png"
    image.save(save_path)


def save_image_with_content_style(save_dir, image, content_image_pil, content_image_path, style_image_path, resolution):
    
    new_image = Image.new('RGB', (resolution*3, resolution))
    if content_image_pil is not None:
        content_image = content_image_pil
    else:
        content_image = Image.open(content_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    style_image = Image.open(style_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)

    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(image, (resolution*2, 0))

    save_path = f"{save_dir}/out_with_cs.jpg"
    new_image.save(save_path)


def x0_from_epsilon(scheduler, noise_pred, x_t, timesteps):
    """Return the x_0 from epsilon
    """
    batch_size = noise_pred.shape[0]
    for i in range(batch_size):
        noise_pred_i = noise_pred[i]
        noise_pred_i = noise_pred_i[None, :]
        t = timesteps[i]
        x_t_i = x_t[i]
        x_t_i = x_t_i[None, :]

        pred_original_sample_i = scheduler.step(
            model_output=noise_pred_i,
            timestep=t,
            sample=x_t_i,
            # predict_epsilon=True,
            generator=None,
            return_dict=True,
        ).pred_original_sample
        if i == 0:
            pred_original_sample = pred_original_sample_i
        else:
            pred_original_sample = torch.cat((pred_original_sample, pred_original_sample_i), dim=0)

    return pred_original_sample


def reNormalize_img(pred_original_sample):
    pred_original_sample = (pred_original_sample / 2 + 0.5).clamp(0, 1)
    
    return pred_original_sample


def normalize_mean_std(image):
    transforms_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transforms_norm(image)

    return image


def is_char_in_font(font_path, char):
    try:
        TTFont_font = TTFont(font_path)
    except:
        # Try to load as TTC (collection)
        try:
            TTFont_font = TTFont(font_path, fontNumber=0)
        except:
            return False
            
    cmap = TTFont_font['cmap']
    for subtable in cmap.tables:
        if ord(char) in subtable.cmap:
            return True
    return False


def load_ttf(ttf_path, fsize=128):
    return ImageFont.truetype(ttf_path, fsize)


def ttf2im(font, char, fsize=128):
    try:
        # Create white image
        image = Image.new('RGB', (fsize, fsize), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Calculate offset to center the EM-SQUARE (using a reference char like '国')
        # This ensures punctuation like '，' stays in the correct relative position
        # instead of being centered as a standalone dot.
        ref_char = "国" 
        
        # Get bounding box of reference char
        # (left, top, right, bottom)
        try:
            ref_bbox = font.getbbox(ref_char)
        except:
            # Fallback for older PIL
            ref_bbox = font.getmask(ref_char).getbbox()
            
        if ref_bbox:
            ref_w = ref_bbox[2] - ref_bbox[0]
            ref_h = ref_bbox[3] - ref_bbox[1]
            
            # Calculate offset to center the reference char
            dx = (fsize - ref_w) // 2 - ref_bbox[0]
            dy = (fsize - ref_h) // 2 - ref_bbox[1]
        else:
            dx, dy = 0, 0
            
        # Draw the actual char using the reference offset
        draw.text((dx, dy), char, font=font, fill=(0, 0, 0))
        
        return image
        
    except Exception as e:
        print(f"Error generating char {char}: {e}")
        return None
