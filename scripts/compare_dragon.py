import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Import synthesis logic to get standard skeleton and generated result
sys.path.append(os.path.join(os.path.dirname(__file__)))
from synthesize_stroke_sequence import StrokeSynthesizer
from optimize_char_flow import optimize_char

def create_comparison(char='龙'):
    # 1. Load Style Image (Yan Style Background) - Light Red
    style_img_path = f"data_examples/train/TargetImage/MyStyle/MyStyle+{char}.jpg"
    
    if os.path.exists(style_img_path):
        bg_img = Image.open(style_img_path).convert('RGBA')
        bg_img = bg_img.resize((512, 512), Image.BILINEAR)
        
        # Colorize to Light Red
        # Convert to grayscale first
        gray = bg_img.convert('L')
        # Invert so ink is white, bg is black
        inverted = ImageOps.invert(gray)
        # Create red mask
        red_layer = Image.new('RGBA', (512, 512), (255, 200, 200, 255)) # Light red background?
        # No, user wants ink to be light red.
        # Let's keep background white, ink light red.
        
        # Create an image with white background
        base = Image.new('RGBA', (512, 512), (255, 255, 255, 255))
        
        # Create ink layer (Light Red)
        ink_color = (255, 150, 150, 255) # Light Red
        ink_layer = Image.new('RGBA', (512, 512), ink_color)
        
        # Use inverted gray as mask for ink
        # Where ink is dark in original, gray is low. Inverted is high.
        # We want ink where original is dark.
        mask = ImageOps.invert(gray)
        
        # Composite
        bg_layer = Image.composite(ink_layer, base, mask)
        
    else:
        print(f"Style image not found: {style_img_path}")
        bg_layer = Image.new('RGBA', (512, 512), (255, 255, 255, 255))

    # 2. Get Standard Skeleton (Gray)
    synth = StrokeSynthesizer()
    raw_data = synth.load_hanzi(char)
    raw_strokes = synth.extract_medians(raw_data)
    bounds = synth.get_char_bounds(raw_strokes)
    norm_strokes = [synth.normalize_stroke(s, bounds) for s in raw_strokes]
    
    skeleton_layer = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    draw_ske = ImageDraw.Draw(skeleton_layer)
    
    for s in norm_strokes:
        pts = [(p[0], p[1]) for p in s]
        if len(pts) > 1:
            draw_ske.line(pts, fill=(180, 180, 180, 255), width=4) # Gray

    # 3. Get Generated Result (Black)
    # We need to run the optimization to get the final strokes
    # But optimize_char_flow saves image. We can modify it to return strokes or just read the code logic.
    # To ensure exact match with previous result, we should ideally reuse the strokes.
    # But for now, let's re-run optimization quickly (it's fast) or just modify optimize_char to return strokes.
    # Actually, optimize_char_flow prints to stdout. 
    # Let's import the logic and run it.
    
    # Run optimization (we'll capture the result by modifying optimize_char or just copy-paste logic)
    # Since we imported optimize_char, let's assume it returns nothing but saves file.
    # I will modify optimize_char in memory or just use the synthesis logic here directly with the same parameters?
    # No, optimization has random retry. Result might differ slightly.
    # But user wants "generated dragon".
    
    # Let's run the optimize_char function and intercept the drawing? 
    # Easier: modify optimize_char to return final_strokes.
    # Or just re-implement the optimization loop here simply.
    
    # Re-run optimization
    print("Re-running optimization for overlay...")
    # ... (Copying optimization logic for brevity and access)
    final_strokes = []
    
    # Gates from optimize_char_flow.py
    GATES = {'pos': 18, 'dir': 6, 'shape': 0, 'corr': 20}
    GATES_SHORT = {'pos': 18, 'dir': 3, 'shape': 0, 'corr': 20}
    
    # Load scoring
    from score_stylized import evaluate_stroke, check_fail_gates
    
    for ref_stroke in norm_strokes:
        best_cand = None
        best_score = -1
        
        is_short = False
        if len(ref_stroke) >= 2:
            length = np.linalg.norm(np.array(ref_stroke[-1]) - np.array(ref_stroke[0]))
            if length < 40: is_short = True
        current_gates = GATES_SHORT if is_short else GATES
        
        # Try 5 times
        best_score_info = {}
        for attempt in range(6):
            params = synth.get_style_params()
            if attempt > 0: params['noise_sigma'] *= 1.2
            
            cand_stroke = synth.apply_transform(ref_stroke, params)
            scores = evaluate_stroke(ref_stroke, cand_stroke, params)
            fails = check_fail_gates(scores, current_gates)
            total = scores['total']
            
            if total > best_score:
                best_score = total
                best_cand = cand_stroke
                best_score_info = scores
                
            if len(fails) == 0 and total >= 85:
                break
        
        print(f"Stroke {len(final_strokes)+1} Score: {best_score:.1f} {best_score_info}")
        final_strokes.append(best_cand)

    generated_layer = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    draw_gen = ImageDraw.Draw(generated_layer)
    
    for s in final_strokes:
        if s:
            pts = [(p[0], p[1]) for p in s]
            if len(pts) > 1:
                draw_gen.line(pts, fill=(0, 0, 0, 255), width=4) # Black
                # Caps
                r = 2
                draw_gen.ellipse([pts[0][0]-r, pts[0][1]-r, pts[0][0]+r, pts[0][1]+r], fill=(0,0,0,255))
                draw_gen.ellipse([pts[-1][0]-r, pts[-1][1]-r, pts[-1][0]+r, pts[-1][1]+r], fill=(0,0,0,255))

    # 4. Composite All
    # Order: Background (Red Ink) -> Skeleton (Gray) -> Generated (Black)
    comp = Image.alpha_composite(bg_layer, skeleton_layer)
    comp = Image.alpha_composite(comp, generated_layer)
    
    out_path = "outputs/compare_dragon.png"
    comp.save(out_path)
    print(f"Saved comparison to {out_path}")

if __name__ == "__main__":
    create_comparison('龙')
