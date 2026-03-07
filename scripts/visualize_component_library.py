import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def visualize_library():
    try:
        with open('data/component_library.json', 'r', encoding='utf-8') as f:
            library = json.load(f)
    except Exception as e:
        print(f"Error loading library: {e}")
        return

    # Sort keys for consistent display
    keys = sorted(library.keys())
    
    # Canvas settings
    cell_size = 128
    margin = 10
    cols = 10
    
    # Calculate total rows needed
    total_components = sum(len(v) for v in library.values())
    
    # Create a big image? Or one per category?
    # Let's do one per category row-wise.
    
    # Calculate height
    # Each category needs: Title row + data rows
    total_height = 0
    for k in keys:
        comps = library[k]
        rows = math.ceil(len(comps) / cols)
        total_height += (rows * cell_size) + 40 # 40 for title
        
    width = cols * cell_size
    img = Image.new('RGB', (width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = None
        small_font = None
        
    current_y = 0
    
    for k in keys:
        comps = library[k]
        
        # Draw Title
        draw.rectangle([0, current_y, width, current_y + 30], fill=(240, 240, 240))
        draw.text((10, current_y + 5), f"Type: {k} (Count: {len(comps)})", fill=(0, 0, 0), font=font)
        current_y += 40
        
        for i, comp in enumerate(comps):
            row = i // cols
            col = i % cols
            
            x = col * cell_size
            y = current_y + row * cell_size
            
            # Draw box
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], outline=(220, 220, 220))
            
            # Draw stroke
            # Points are normalized to [-1, 1] range (centered at 0)
            pts = np.array(comp['points'])
            
            # Scale to fit cell (leaving margin)
            # Map [-1, 1] to [margin, cell_size - margin]
            scale = (cell_size - 2 * margin) / 2.0
            pts_screen = pts * scale + [x + cell_size/2, y + cell_size/2]
            
            pts_list = [tuple(p) for p in pts_screen]
            if len(pts_list) > 1:
                draw.line(pts_list, fill=(0, 0, 0), width=2)
                # Mark start
                draw.ellipse([pts_list[0][0]-2, pts_list[0][1]-2, pts_list[0][0]+2, pts_list[0][1]+2], fill=(255, 0, 0))
                
            # Label
            label = f"{comp['char']}[{comp['index']}]"
            draw.text((x + 2, y + 2), label, fill=(100, 100, 100), font=small_font)
            
        # Advance Y
        rows = math.ceil(len(comps) / cols)
        current_y += rows * cell_size
        
    out_path = "outputs/component_library_viz.png"
    img.save(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    visualize_library()
