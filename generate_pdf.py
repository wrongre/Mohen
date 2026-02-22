
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import cv2
import numpy as np

def create_template(filename):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    # Register Fonts
    font_medium = "ChineseMedium"
    font_thin = "ChineseThin"
    
    # Paths to check (Updated to include AlibabaPuHuiTi 3.0 paths from ls output)
    medium_paths = [
        "web_ui/static/fonts/AlibabaPuHuiTi-3-65-Medium.ttf",
        "web_ui/static/fonts/Alibaba-PuHuiTi-Medium.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf"
    ]
    
    thin_paths = [
        "web_ui/static/fonts/AlibabaPuHuiTi-3-45-Light.ttf", # Using Light for Thin equivalent
        "web_ui/static/fonts/Alibaba-PuHuiTi-Light.ttf",
        "C:\\Windows\\Fonts\\msyhl.ttc",
        "C:\\Windows\\Fonts\\simsun.ttc"
    ]
    
    def register_font(name, paths):
        for p in paths:
            if os.path.exists(p):
                try:
                    pdfmetrics.registerFont(TTFont(name, p))
                    print(f"Loaded font {name} from {p}")
                    return True
                except Exception as e:
                    print(f"Failed to load {p}: {e}")
        return False

    if not register_font(font_medium, medium_paths):
        print("Warning: No suitable Medium font found. Using Helvetica.")
        font_medium = "Helvetica"
        
    if not register_font(font_thin, thin_paths):
        print("Warning: No suitable Thin font found. Using Helvetica.")
        font_thin = "Helvetica"

    # Characters to fill (30 chars)
    chars = [
        '永', '十', '一', '乙', '水', '木',
        '国', '回', '幽', '巫', '用', '月',
        '飞', '也', '戈', '身', '走', '这',
        '神', '韵', '龙', '舞', '灵', '繁', '墨', '魂', '德', '和', '美', '气'
    ]

    # --- Header ---
    c.setFont(font_medium, 14)
    # Check for logo
    logo_path = "figures/logo.png"
    if os.path.exists(logo_path):
        # Draw Logo (Height ~ 15mm, preserve aspect ratio)
        # Position: Left aligned, vertically centered in header space
        logo_h = 12*mm
        # Assuming logo is square-ish or wide. Let's just set height and let width scale if needed, 
        # but reportlab drawImage needs w/h. Let's assume square or get size.
        # Simple approach: draw image
        try:
            c.drawImage(logo_path, 20*mm, height - 25*mm, height=logo_h, width=logo_h*2.5, preserveAspectRatio=True, mask='auto')
        except:
            c.drawString(20*mm, height - 20*mm, "墨痕 | MoHen")
    else:
        c.drawString(20*mm, height - 20*mm, "墨痕 | MoHen")
    
    # --- ArUco Markers ---
    # Layout: 8 columns x 4 rows
    cols = 8
    rows = 4
    grid_w = 18*mm
    gap = 2*mm
    total_grid_w = cols * grid_w + (cols - 1) * gap
    start_x = (width - total_grid_w) / 2
    
    # Start Y: Below header
    start_y = height - 40*mm
    total_grid_h = rows * grid_w + (rows - 1) * gap
    
    # Generate 4 markers (DICT_4X4_50, IDs 0, 1, 2, 3)
    # User Request: Smaller size, better alignment, softer color.
    marker_size = 8*mm # Reduced from 12mm
    marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Gap between grid and marker
    marker_gap = 4*mm
    
    # Define positions (Bottom-Left corner of the image for reportlab)
    
    # TL Marker (ID 0): Left of "永" (Grid TL)
    # Vertical: Aligned with "为了..." prompt line.
    # Prompt is at start_y + 5mm.
    # Let's align Marker Center with Prompt Baseline (approx).
    # Marker Bottom = start_y + 1mm. Marker Center = start_y + 5mm.
    pos_tl = (start_x - marker_size - marker_gap, start_y + 1*mm)
    
    # TR Marker (ID 1): Right of "回" (Grid TR)
    # Vertical: Same as TL
    pos_tr = (start_x + total_grid_w + marker_gap, start_y + 1*mm)
    
    # BR Marker (ID 2): Right of last cell (Grid BR)
    # Vertical: Top aligned with Grid Bottom.
    # Grid Bottom Y = start_y - total_grid_h
    # Marker Top Y = start_y - total_grid_h
    # Marker Bottom Y = start_y - total_grid_h - marker_size
    pos_br = (start_x + total_grid_w + marker_gap, start_y - total_grid_h - marker_size)
    
    # BL Marker (ID 3): Left of "墨" (Grid BL)
    # Vertical: Same as BR
    pos_bl = (start_x - marker_size - marker_gap, start_y - total_grid_h - marker_size)
    
    markers_list = [
        (0, pos_tl),
        (1, pos_tr),
        (2, pos_br),
        (3, pos_bl)
    ]
    
    if not os.path.exists("tmp_markers"):
        os.makedirs("tmp_markers")
        
    for mid, (mx, my) in markers_list:
        # Generate marker
        img = cv2.aruco.generateImageMarker(marker_dict, mid, 200) # 200x200 px
        
        # Soften Color: Change Black (0) to Dark Grey (80)
        # This reduces visual harshness while maintaining detection contrast
        img[img == 0] = 80 
        
        tmp_path = f"tmp_markers/marker_{mid}.png"
        cv2.imwrite(tmp_path, img)
        
        # Draw on PDF
        c.drawImage(tmp_path, mx, my, width=marker_size, height=marker_size)
        
    # --- Grid Area ---
    # (Dimensions defined above)
    
    c.setStrokeColorRGB(0.6, 0.6, 0.6) # Light gray markers
    c.setLineWidth(0.5)
    
    # Grid Prompt (Moved to top-left of grid area)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.setFont(font_thin, 10)
    # Position: align left with grid, slightly above
    c.drawString(start_x, start_y + 5*mm, "为了保证生成质量，请尽量将这里的文字都写出来")

    idx = 0
    marker_len = 3*mm # Length of L-marker legs
    
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * (grid_w + gap)
            y = start_y - row * (grid_w + gap) - grid_w # Bottom-left
            
            # Draw Dashed Box (Light Grey - Optimized for Detection)
            c.setStrokeColorRGB(0.6, 0.6, 0.6) # Darker grey to ensure detection
            c.setLineWidth(0.5)
            c.setDash([4, 2]) # Longer dashes, shorter gaps (easier to connect)
            
            c.rect(x, y, grid_w, grid_w)
            
            c.setDash([]) # Reset to solid for text/other items
            
            # Draw Guide Character (Small, Top-Left)
            if idx < len(chars):
                char = chars[idx]
                c.setFillColorRGB(0.7, 0.7, 0.7) # Lighter grey
                c.setFont(font_medium, 8) # Small font
                
                # Position: Top-Left inside padding
                text_x = x + 2*mm
                text_y = y + grid_w - 4*mm
                
                c.drawString(text_x, text_y, char)
                idx += 1
            else:
                pass
    
    # --- Flow Area ---
    flow_start_y = height - 150*mm
    margin = 20*mm
    line_height = 12*mm
    line_w = width - 2 * margin
    
    c.setStrokeColorRGB(0.8, 0.8, 0.8) # Light gray lines
    c.setLineWidth(1)
    
    for i in range(8):
        y = flow_start_y - i * line_height
        c.line(margin, y, margin + line_w, y)
        
    # Flow Prompt
    prompt_y = flow_start_y + 12*mm
    
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.setFont(font_thin, 10)
    c.drawString(margin, prompt_y + 2*mm, "下面的横格上可以按你的想法写下你想写的内容，比如一首诗，一段话都可以，按照你平时的书写习惯写就可以。")
    
    c.save()
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    create_template("web_ui/static/MoHen_Template_V1.pdf")
