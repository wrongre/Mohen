import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
# Import our new modules
sys.path.append(os.path.join(os.path.dirname(__file__)))
from synthesize_stroke_sequence import StrokeSynthesizer
from score_stylized import evaluate_stroke, check_fail_gates

def optimize_char(char, max_retries=5):
    synth = StrokeSynthesizer()
    raw_data = synth.load_hanzi(char)
    raw_strokes = synth.extract_medians(raw_data)
    
    if not raw_strokes:
        print(f"No data for {char}")
        return
        
    bounds = synth.get_char_bounds(raw_strokes)
    norm_strokes = [synth.normalize_stroke(s, bounds) for s in raw_strokes]
    
    # Flip Y for visualization (since we use image coords)
    # Actually normalize_stroke already mapped to image coords?
    # Let's check: miny maps to margin.
    # If raw data has y up (standard math), we need to flip.
    # makemeahanzi usually has y down (svg screen coords).
    # We will assume y down for now.
    
    final_strokes = []
    history = []
    
    print(f"Optimizing '{char}' ({len(norm_strokes)} strokes)...")
    
    # Define Gates
    GATES = {'pos': 18, 'dir': 6, 'shape': 0, 'corr': 20}
    # Short stroke relaxation
    GATES_SHORT = {'pos': 18, 'dir': 3, 'shape': 0, 'corr': 20}
    
    for idx, ref_stroke in enumerate(norm_strokes):
        print(f"  Stroke {idx+1}:", end="")
        
        best_cand = None
        best_score = -1
        accepted = False
        
        # Determine if short stroke
        is_short = False
        if len(ref_stroke) >= 2:
            length = np.linalg.norm(np.array(ref_stroke[-1]) - np.array(ref_stroke[0]))
            if length < 40: is_short = True
        
        current_gates = GATES_SHORT if is_short else GATES
        
        # Optimization Loop
        for attempt in range(max_retries + 1):
            # 1. Sample Params
            params = synth.get_style_params()
            
            # If retrying, maybe adjust params based on previous failure?
            # Simple random search for now
            if attempt > 0:
                params['noise_sigma'] *= 1.2 # Increase chaos
                
            # 2. Generate
            cand_stroke = synth.apply_transform(ref_stroke, params)
            
            # 3. Score
            scores = evaluate_stroke(ref_stroke, cand_stroke, params)
            fails = check_fail_gates(scores, current_gates)
            
            is_pass = (len(fails) == 0)
            total = scores['total']
            
            # Record
            if total > best_score:
                best_score = total
                best_cand = (cand_stroke, params, scores)
            
            # Check Threshold
            if is_pass and total >= 85:
                print(f" ACCEPT ({total:.1f})")
                accepted = True
                break
            
            if attempt < max_retries:
                print(f" .", end="")
            else:
                print(f" FAIL ({total:.1f}) -> Best: {best_score:.1f} {fails}")
        
        # Lock best result
        final_strokes.append(best_cand[0])
        history.append({
            'stroke_idx': idx,
            'accepted': accepted,
            'score': best_cand[2],
            'params': best_cand[1]
        })

    # Render Final
    img = Image.new('RGB', (512, 512), (255,255,255))
    draw = ImageDraw.Draw(img)
    
    # Draw Reference (Gray)
    for s in norm_strokes:
        pts = [(float(p[0]), float(p[1])) for p in s]
        draw.line(pts, fill=(200,200,200), width=2)
        
    # Draw Optimized (Black)
    for i, s in enumerate(final_strokes):
        if not s: continue
        pts = [(float(p[0]), float(p[1])) for p in s]
        draw.line(pts, fill=(0,0,0), width=4)
        # draw caps
        if pts:
            r = 2
            draw.ellipse([pts[0][0]-r, pts[0][1]-r, pts[0][0]+r, pts[0][1]+r], fill=0)
            draw.ellipse([pts[-1][0]-r, pts[-1][1]-r, pts[-1][0]+r, pts[-1][1]+r], fill=0)
            
            # Draw score
            score = history[i]['score']['total']
            draw.text((pts[0][0], pts[0][1]-10), f"{score:.1f}", fill=(0,0,255))

    out_path = f"outputs/opt_{char}.png"
    img.save(out_path)
    print(f"Saved visualization to {out_path}")
    
    # Generate Concat
    concat_imgs = []
    for i in range(1, len(final_strokes) + 1):
        step_img = Image.new('RGB', (512, 512), (255, 255, 255))
        step_draw = ImageDraw.Draw(step_img)
        # Draw strokes up to i
        for j in range(i):
            s = final_strokes[j]
            if not s: continue
            pts = [(float(p[0]), float(p[1])) for p in s]
            step_draw.line(pts, fill=(0,0,0), width=4)
            if pts:
                r = 2
                step_draw.ellipse([pts[0][0]-r, pts[0][1]-r, pts[0][0]+r, pts[0][1]+r], fill=0)
                step_draw.ellipse([pts[-1][0]-r, pts[-1][1]-r, pts[-1][0]+r, pts[-1][1]+r], fill=0)
                
                # Draw score for current stroke
                if j == i - 1:
                    score = history[j]['score']['total']
                    step_draw.text((pts[0][0], pts[0][1]-10), f"{score:.1f}", fill=(0,0,255))
        concat_imgs.append(step_img)
    
    # Stitch horizontally
    total_w = 512 * len(concat_imgs)
    concat_final = Image.new('RGB', (total_w, 512), (255, 255, 255))
    for i, im in enumerate(concat_imgs):
        concat_final.paste(im, (i * 512, 0))
    
    concat_path = f"outputs/opt_{char}_concat.png"
    concat_final.save(concat_path)
    print(f"Saved concat visualization to {concat_path}")
    
    # Save report
    with open(f"outputs/opt_{char}_report.json", 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    optimize_char('龙')
