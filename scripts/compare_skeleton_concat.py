import os
from pathlib import Path
from PIL import Image
import numpy as np
import math


def find_skeleton(char):
    base = Path('outputs/synth_samples')
    candidates = [base / f"{char}_skeleton_flipped.png", base / f"{char}_skeleton.png", base / f"{char}_skeleton_rotated.png"]
    for p in candidates:
        if p.exists():
            return str(p)
    # fallback
    p = base / f"{char}_skeleton.png"
    return str(p)


def find_concat(char):
    p = Path('outputs/synth_samples') / f"{char}_strokes_concat.png"
    if p.exists():
        return str(p)
    # fallback: try concatenation of steps
    return None


def image_stats(img):
    arr = np.array(img.convert('L'))
    # threshold by mean
    thr = arr.mean()
    ink = (arr < thr).astype(np.uint8)
    h,w = ink.shape
    ink_pixels = int(ink.sum())
    ink_ratio = ink_pixels / (h*w)
    coords = np.column_stack(np.nonzero(ink))
    if coords.size == 0:
        centroid = None
        bbox = None
        orientation = None
        comps = 0
    else:
        centroid = (float(coords[:,1].mean()), float(coords[:,0].mean()))
        bbox = [int(coords[:,1].min()), int(coords[:,0].min()), int(coords[:,1].max()), int(coords[:,0].max())]
        pts = coords - coords.mean(axis=0)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, np.argmax(eigvals)]
        orientation = math.degrees(math.atan2(major[0], major[1]))
        # connected components via simple flood fill
        comps = 0
        try:
            import cv2
            num_labels, labels = cv2.connectedComponents(ink.astype('uint8'))
            comps = int(num_labels - 1)
        except Exception:
            # approximate by counting disconnected bounding boxes on coarse grid
            comps = 1

    return {
        'ink_ratio': ink_ratio,
        'centroid': centroid,
        'bbox': bbox,
        'orientation': orientation,
        'components': comps,
        'size': (w,h)
    }


def crop_last_column(concat_path, steps_dir):
    # determine number of steps by listing steps_dir
    step_dir = Path(steps_dir)
    if step_dir.exists():
        steps = sorted([p for p in step_dir.iterdir() if p.name.startswith('step_') and p.suffix.lower() in ['.png','.jpg']])
        n = len(steps)
    else:
        n = None

    img = Image.open(concat_path)
    W, H = img.size
    if n is None or n <= 1:
        # return whole image
        return img
    col_w = W // n
    box = (W - col_w, 0, W, H)
    return img.crop(box)


def human_report(a_stats, b_stats):
    lines = []
    lines.append(f"骨架 墨量={a_stats['ink_ratio']:.4f}, 拼接最后帧 墨量={b_stats['ink_ratio']:.4f}")
    lines.append(f"骨架 连通分量={a_stats['components']}, 拼接最后帧 连通分量={b_stats['components']}")
    lines.append(f"骨架 包围盒={a_stats['bbox']}, 拼接最后帧 包围盒={b_stats['bbox']}")
    lines.append(f"骨架 方向={a_stats['orientation']}, 拼接最后帧 方向={b_stats['orientation']}")
    lines.append("")
    if b_stats['ink_ratio'] > a_stats['ink_ratio'] * 0.6:
        lines.append("结论：拼接图看起来不是纯空心—它的墨量明显高于骨架，但若内部仍显白色，说明笔划只是描边而未填充内部（造成‘空心’感）。")
    else:
        lines.append("结论：拼接图墨量接近骨架，若视觉上是空心，可能是线条仅绘制轮廓或线宽不足以填满笔画间隙。")
    lines.append("建议：若希望实心效果，可将中线扩成笔宽填充（distance transform 或 stroke buffering），或在绘制时用较粗笔与端点圆来封闭空隙。")
    return "\n".join(lines)


def main(char='龙'):
    sk = find_skeleton(char)
    concat = find_concat(char)
    if not os.path.exists(sk):
        print('Skeleton not found:', sk)
        return 2
    if not concat or not os.path.exists(concat):
        print('Concat not found:', concat)
        return 3

    last = crop_last_column(concat, os.path.join('outputs', 'synth_samples', f'{char}_stroke_steps'))

    a_stats = image_stats(Image.open(sk))
    b_stats = image_stats(last)

    print('Skeleton:', sk)
    print(a_stats)
    print('\nConcat last column stats:')
    print(b_stats)
    print('\n---- Conclusion ----')
    print(human_report(a_stats, b_stats))


if __name__ == '__main__':
    import sys
    ch = sys.argv[1] if len(sys.argv) > 1 else '龙'
    main(ch)
