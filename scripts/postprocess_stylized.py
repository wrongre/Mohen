import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps


def enhance_step_image(p: Path, dilate_iters=2, blur_radius=1):
    im = Image.open(p).convert('L')
    # Binarize by mean
    thr = im.point(lambda x: 0 if x < im.convert('L').point(lambda y: y).histogram().__len__() else 255)
    # Simpler: use adaptive threshold via point with mean
    arr = im
    mean = sum(im.getdata()) / (im.width * im.height)
    bw = im.point(lambda x: 0 if x < mean else 255)
    # Invert so ink=255 for dilation
    inv = ImageOps.invert(bw)
    # Dilate via MaxFilter multiple times
    for i in range(dilate_iters):
        inv = inv.filter(ImageFilter.MaxFilter(3))
    # Blur slightly to get brushy edges
    inv = inv.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # Re-invert back to black-on-white
    out = ImageOps.invert(inv).convert('RGB')
    return out


def main(char='龙', steps_dir=None, out_dir='outputs/synth_samples'):
    if steps_dir is None:
        steps_dir = os.path.join(out_dir, f"{char}_stylized_steps")
    p = Path(steps_dir)
    if not p.exists():
        print('Steps dir not found:', steps_dir)
        return 2
    imgs = sorted([x for x in p.iterdir() if x.suffix.lower() in ['.png','.jpg']])
    enhanced_paths = []
    enh_dir = p.parent / f"{char}_stylized_steps_enh"
    enh_dir.mkdir(parents=True, exist_ok=True)
    for img_path in imgs:
        out_im = enhance_step_image(img_path, dilate_iters=3, blur_radius=1.2)
        out_path = enh_dir / img_path.name.replace('step_','step_enh_')
        out_im.save(out_path)
        enhanced_paths.append(str(out_path))
    # concat enhanced
    from PIL import Image
    imgs_pil = [Image.open(x).convert('RGB') for x in enhanced_paths]
    widths, heights = zip(*(i.size for i in imgs_pil))
    total_w = sum(widths)
    max_h = max(heights)
    new_im = Image.new('RGB', (total_w, max_h), (255,255,255))
    x = 0
    for im in imgs_pil:
        new_im.paste(im, (x,0))
        x += im.width
    concat_path = Path(out_dir) / f"{char}_stylized_concat_enh.png"
    new_im.save(concat_path)
    print('Saved enhanced concat:', concat_path)
    for p in enhanced_paths:
        print('-', p)
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--char', type=str, default='龙')
    args = parser.parse_args()
    main(args.char)
