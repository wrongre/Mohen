import os
import sys
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:
    cv2 = None


# reuse simple threshold and thinning from extractor
from extract_style_profile import otsu_thresh, zhang_suen_thinning


def load_profile(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_skeleton_points(img_path, resize=256):
    img = Image.open(img_path).convert('L')
    img = img.resize((resize, resize), Image.BILINEAR)
    binary = otsu_thresh(img)
    ske = zhang_suen_thinning(binary)
    pts = np.column_stack(np.nonzero(ske))
    return pts, img.size, img


def apply_style_transform(pts, img_size, scale=(1.0,1.0), slant=0.0, jitter=0.0):
    # pts are (row, col) coordinates; convert to (x,y) with origin center
    h,w = img_size[1], img_size[0]
    coords = pts.astype(float)
    # swap to x,y = col,row
    xy = np.stack([coords[:,1], coords[:,0]], axis=1)
    center = np.array([w/2.0, h/2.0])
    rel = xy - center
    # apply slant: shear in x by factor = tan(slant_rad)
    slant_rad = math.radians(slant)
    shear = math.tan(slant_rad)
    # apply scale
    sx, sy = scale
    transformed = np.empty_like(rel)
    transformed[:,0] = (rel[:,0] + rel[:,1]*shear) * sx
    transformed[:,1] = rel[:,1] * sy
    # jitter
    if jitter > 0:
        noise = np.random.normal(scale=jitter, size=transformed.shape)
        transformed += noise
    out = transformed + center
    return out


def render_points_to_image(points, img_size, pen_width=3, out_path=None):
    w,h = img_size
    canvas = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    r = max(1, int(pen_width/2))
    # draw circles for each point
    for (x,y) in points:
        bbox = [x-r, y-r, x+r, y+r]
        draw.ellipse(bbox, fill=(0,0,0))
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        canvas.save(out_path)
    return canvas


def synth_one(char_filename, profile_path, out_dir='outputs/synth_samples'):
    profile = load_profile(profile_path)
    agg = profile.get('aggregate', {})
    pen_w = agg.get('pen_width', {}).get('mean', 3.0)
    pen_std = agg.get('pen_width', {}).get('std', 0.8)
    slant_deg = agg.get('slant_deg', {}).get('mean', 0.0) if agg.get('slant_deg') else 0.0
    # choose scale from curvature (heuristic)
    curv = agg.get('curvature_mean_deg', {}).get('mean', 20.0) if agg.get('curvature_mean_deg') else 20.0
    scale_x = 1.0
    scale_y = 1.0
    # map curvature to slight vertical compression if high curvature
    scale_y = max(0.8, 1.0 - (min(curv, 60) / 200.0))
    base_jitter = pen_std * 0.6

    pts, img_size, img = get_skeleton_points(char_filename, resize=256)
    if len(pts) == 0:
        print('No skeleton points for', char_filename)
        return None
    styled = apply_style_transform(pts, img_size, scale=(scale_x, scale_y), slant=slant_deg, jitter=base_jitter)
    out_name = Path(char_filename).stem + '_stylized.png'
    out_path = os.path.join(out_dir, out_name)
    canvas = render_points_to_image(styled, img_size, pen_width=max(1, pen_w), out_path=out_path)
    print('Saved', out_path)
    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, default='data_examples/style_profiles/MyStyle.json')
    parser.add_argument('--char', type=str, default='data_examples/train/ContentImage/龙.jpg')
    parser.add_argument('--out_dir', type=str, default='outputs/synth_samples')
    args = parser.parse_args()

    synth_one(args.char, args.profile, args.out_dir)
