
import os
import shutil
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path

def ttf2im(char, font_path, size=96):
    try:
        # Check if ttc and handle collection
        if font_path.lower().endswith(".ttc"):
             font = ImageFont.truetype(font_path, size=int(size * 0.8), index=0)
        else:
             font = ImageFont.truetype(font_path, size=int(size * 0.8))
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        return None
    
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Calculate offset to center the EM-SQUARE (using a reference char like '国')
    # This matches the logic in utils.py for inference consistency
    ref_char = "国"
    try:
        ref_bbox = font.getbbox(ref_char)
    except:
        ref_bbox = font.getmask(ref_char).getbbox()
        
    if ref_bbox:
        ref_w = ref_bbox[2] - ref_bbox[0]
        ref_h = ref_bbox[3] - ref_bbox[1]
        
        dx = (size - ref_w) // 2 - ref_bbox[0]
        dy = (size - ref_h) // 2 - ref_bbox[1]
    else:
        dx, dy = 0, 0
    
    # Draw
    draw.text((dx, dy), char, font=font, fill=(0, 0, 0))
    return img

def prepare_data():
    project_root = Path(os.getcwd())
    
    # Source
    processed_dir = project_root / "data_examples/processed"
    
    # Destination
    train_content_dir = project_root / "data_examples/train/ContentImage"
    train_target_dir = project_root / "data_examples/train/TargetImage"
    my_style_dir = train_target_dir / "MyStyle"
    
    # Create directories
    os.makedirs(train_content_dir, exist_ok=True)
    os.makedirs(my_style_dir, exist_ok=True)
    
    # Fonts
    content_font_path = "C:/Windows/Fonts/simsun.ttc"
    neg_font_path = "C:/Windows/Fonts/simhei.ttf"
    
    if not os.path.exists(content_font_path):
        print(f"Warning: Content font {content_font_path} not found. Trying fallback.")
        # Try to find any font in web_ui/static/fonts
        fonts = list((project_root / "web_ui/static/fonts").glob("*.ttf"))
        if fonts:
            content_font_path = str(fonts[0])
            
    print(f"Using Content Font: {content_font_path}")
    
    processed_count = 0
    
    # Iterate processed images
    if not processed_dir.exists():
        print("Error: No processed images found (directory missing).")
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    files = os.listdir(processed_dir)
    print(f"Found {len(files)} files in {processed_dir}")
    
    for file_name in files:
        if not file_name.endswith(".jpg") or "grid_warped" in file_name:
            continue
            
        # Format: {idx}_{char}.jpg
        parts = file_name.split('_')
        if len(parts) < 2:
            continue
            
        char = parts[1].replace(".jpg", "")
        print(f"Processing char: {char}")
        
        # 1. Target Image (MyStyle)
        src_path = processed_dir / file_name
        dst_name = f"MyStyle+{char}.jpg"
        dst_path = my_style_dir / dst_name
        
        img = Image.open(src_path).convert('RGB')
        img = img.resize((96, 96)) # Resize to model input size
        img.save(dst_path)
        
        # 2. Content Image (SimSun)
        content_img_name = f"{char}.jpg"
        content_img_path = train_content_dir / content_img_name
        
        # Always regenerate to ensure consistency
        content_img = ttf2im(char, content_font_path)
        if content_img:
            content_img.save(content_img_path)
        
        # 3. Negative Sample (SimHei)
        if os.path.exists(neg_font_path):
            neg_style_dir = train_target_dir / "SimHei"
            os.makedirs(neg_style_dir, exist_ok=True)
            neg_img_name = f"SimHei+{char}.jpg"
            neg_img_path = neg_style_dir / neg_img_name
            
            neg_img = ttf2im(char, neg_font_path)
            if neg_img:
                neg_img.save(neg_img_path)
                
        processed_count += 1
        
    print(f"Prepared {processed_count} samples for training.")
    if processed_count == 0:
        raise ValueError("No valid samples found in processed directory! Please check Step 2.")

if __name__ == "__main__":
    prepare_data()
