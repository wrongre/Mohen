import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_slice import process_scanned_template, crop_and_save
from pydantic import BaseModel

app = FastAPI()

class CropRequest(BaseModel):
    index: int
    x: int
    y: int
    w: int
    h: int

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# Ensure directories exist
os.makedirs("web_ui/static", exist_ok=True)
os.makedirs("web_ui/templates", exist_ok=True)

# Ensure upload directory exists
UPLOAD_DIR = "data_examples/upload"
PROCESSED_DIR = "data_examples/processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="web_ui/static"), name="static")
app.mount("/sliced_images", StaticFiles(directory="data_examples/processed"), name="sliced_images")

# Global store for processed chars (in memory for now, DB later)
PROCESSED_CHARS = []

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Do NOT clear session data on visit - only clear explicitly via API if needed
    # This preserves state if user navigates back accidentally
    return read_template("step1.html")

@app.get("/step2", response_class=HTMLResponse)
async def read_step2():
    return read_template("step2.html")

@app.get("/step3", response_class=HTMLResponse)
async def read_step3():
    return read_template("step3.html")

@app.get("/generate", response_class=HTMLResponse)
async def read_generate():
    return read_template("generate.html")

def read_template(filename):
    try:
        with open(f"web_ui/templates/{filename}", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Template {filename} not found."

import subprocess
import threading
import time

TRAINING_STATUS = {
    "state": "idle", # idle, preparing, training, completed, failed
    "progress": 0,
    "log": [],
    "error": None
}

def run_training_task():
    global TRAINING_STATUS
    TRAINING_STATUS["state"] = "preparing"
    TRAINING_STATUS["progress"] = 0
    TRAINING_STATUS["log"] = ["Starting preparation..."]
    
    try:
        # 0. Clean up previous training outputs to force fresh training
        import shutil
        output_dir = "outputs/fine_tuning"
        if os.path.exists(output_dir):
            TRAINING_STATUS["log"].append("Cleaning up previous training data...")
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(f"Warning: Failed to clean output dir: {e}")
        
        # 1. Prepare Dataset
        process = subprocess.Popen(
            ["python", "scripts/prepare_dataset.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Preparation failed: {stderr}")
        TRAINING_STATUS["log"].append("Dataset preparation complete.")
        TRAINING_STATUS["progress"] = 5 # Set to 5% after prep
        
        # 2. Start Training
        TRAINING_STATUS["state"] = "training"
        TRAINING_STATUS["log"].append("Starting training process...")
        
        # Construct command
        # Using python -m accelerate.commands.launch for better Windows compatibility
        cmd = [
            "python", "-m", "accelerate.commands.launch", "train.py",
            "--seed=123",
            "--experience_name=FontDiffuser_fine_tuning",
            "--data_root=data_examples",
            "--output_dir=outputs/fine_tuning",
            "--report_to=tensorboard",
            "--phase_2",
            "--phase_1_ckpt_dir=ckpt",
            "--scr_ckpt_path=ckpt/scr_210000.pth",
            "--sc_coefficient=0.1", # Increased from 0.01 to 0.1 to force stronger content adherence
            "--num_neg=16",
            "--resolution=96",
            "--style_image_size=96",
            "--content_image_size=96",
            "--content_encoder_downsample_size=3",
            "--channel_attn=True",
            "--content_start_channel=64",
            "--style_start_channel=64",
            "--train_batch_size=4",
            "--perceptual_coefficient=0.1", 
            "--offset_coefficient=0.5",
            "--max_train_steps=1500", 
            "--ckpt_interval=500",
            "--gradient_accumulation_steps=1",
            "--log_interval=10",
            "--learning_rate=1e-6", # Reduced from 5e-6 to 1e-6 to be extra careful with structure
            "--lr_scheduler=constant",
            "--lr_warmup_steps=100",
            "--drop_prob=0.1",
            "--mixed_precision=no"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr to stdout to see errors
            text=True,
            cwd=os.getcwd(),
            bufsize=1 # Line buffered
        )
        
        # Monitor logs
        total_steps = 2000
        full_log = []
        while True:
            # Read line by line
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(f"Train Log: {line.strip()}") # Print to console too
                TRAINING_STATUS["log"].append(line.strip())
                full_log.append(line.strip())
                # Parse progress
                # Handle tqdm format: "Steps:  30%|...| 603/2000 ..."
                if "|" in line and "/" in line and "Steps:" in line:
                    try:
                        # Extract "603/2000" part
                        parts = line.split("|")[-1].strip().split(" ")[0] # Should get 603/2000
                        if "/" in parts:
                            current, total = map(int, parts.split("/"))
                            TRAINING_STATUS["progress"] = 5 + int((current / total) * 95)
                    except:
                        pass
                
                # Handle manual log format: "Global Step 10 => ..."
                elif "Global Step" in line:
                    try:
                        step = int(line.split("Global Step")[1].split("=>")[0].strip())
                        # Map 0-total_steps to 5-100
                        TRAINING_STATUS["progress"] = 5 + int((step / total_steps) * 95)
                    except:
                        pass
        
        # Write debug log
        try:
            with open("debug_training_last.log", "w", encoding="utf-8") as f:
                f.write("\n".join(full_log))
        except:
            pass
            
        if process.returncode != 0:
            # Try to read remaining stderr if any
            stderr = process.stdout.read() # We merged stderr to stdout
            raise Exception(f"Training failed with code {process.returncode}: {stderr}")
            
        # Heuristic check: Did we actually train?
        # Check if "Global Step" ever appeared or if log is too short
        if len(full_log) < 10 or not any("Global Step" in l for l in full_log):
             # Suspiciously short run
             raise Exception("Training process exited prematurely. Check logs.")
            
        TRAINING_STATUS["state"] = "completed"
        TRAINING_STATUS["progress"] = 100
        TRAINING_STATUS["log"].append("Training completed successfully.")
        
        # Post-Training: Backup reference images to fine_tuning output
        # This ensures "Current Training Session" is self-contained even if processed dir is cleared
        try:
            import shutil
            ref_backup_dir = os.path.join("outputs/fine_tuning", "reference_images")
            os.makedirs(ref_backup_dir, exist_ok=True)
            
            if os.path.exists(PROCESSED_DIR):
                processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".jpg") and "grid" not in f]
                # Backup up to 50 images to be safe
                for fname in processed_files[:50]:
                    shutil.copy2(os.path.join(PROCESSED_DIR, fname), os.path.join(ref_backup_dir, fname))
                
                grid_src = os.path.join(PROCESSED_DIR, "grid_warped.jpg")
                if os.path.exists(grid_src):
                    shutil.copy2(grid_src, os.path.join(ref_backup_dir, "grid_warped.jpg"))
                    
            TRAINING_STATUS["log"].append("Reference images backed up.")
        except Exception as e:
            print(f"Warning: Failed to backup images: {e}")
        
    except Exception as e:
        TRAINING_STATUS["state"] = "failed"
        TRAINING_STATUS["error"] = str(e)
        TRAINING_STATUS["log"].append(f"Error: {e}")
        # Also write debug log on error
        try:
            with open("debug_training_last.log", "a", encoding="utf-8") as f:
                f.write(f"\n\nEXCEPTION: {e}")
        except:
            pass

@app.post("/api/train")
async def start_training():
    global TRAINING_STATUS
    if TRAINING_STATUS["state"] in ["preparing", "training"]:
        return JSONResponse(content={"message": "Training already in progress"}, status_code=409)
    
    # Reset status
    TRAINING_STATUS["state"] = "preparing"
    TRAINING_STATUS["progress"] = 0
    TRAINING_STATUS["log"] = ["Starting preparation..."]
    TRAINING_STATUS["error"] = None
    
    # Synchronously clean up previous training outputs
    # This ensures that polling won't find old files
    import shutil
    output_dir = "outputs/fine_tuning"
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            TRAINING_STATUS["log"].append("Cleaned up previous training data.")
        except Exception as e:
            print(f"Warning: Failed to clean output dir: {e}")
            TRAINING_STATUS["log"].append(f"Warning: Failed to clean output dir: {e}")
            
    # Check if processed data exists
    if not os.path.exists(PROCESSED_DIR):
        return JSONResponse(content={"message": "Processed directory not found."}, status_code=400)
        
    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".jpg") and "grid" not in f]
    if len(processed_files) == 0:
        return JSONResponse(content={"message": "No training data found. Please upload and process a template in Step 1 & 2."}, status_code=400)
        
    thread = threading.Thread(target=run_training_task)
    thread.start()
    
    return JSONResponse(content={"message": "Training started"})

import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from src import (FontDiffuserDPMPipeline, FontDiffuserModelDPM, build_ddpm_scheduler,
                 build_unet, build_content_encoder, build_style_encoder)
from utils import ttf2im, is_char_in_font

# Model Cache
MODEL_CACHE = {}

class GenerateRequest(BaseModel):
    text: str
    style_dir: str
    variation: float = 0.5
    density: int = 50
    chunk_index: int = 0

def get_latest_ckpt(base_dir):
    if not os.path.exists(base_dir):
        return None
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("global_step_")]
    if not subdirs:
        return None
    try:
        latest = max(subdirs, key=lambda x: int(x.split("_")[-1]))
        return os.path.join(base_dir, latest)
    except:
        return None

def load_model(style_dir):
    global MODEL_CACHE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if style_dir in MODEL_CACHE:
        return MODEL_CACHE[style_dir]
        
    print(f"Loading model from {style_dir}...")
    
    # Mock args needed for build functions
    class Args:
        def __init__(self):
            self.unet_channels = (64, 128, 256, 512)
            self.style_image_size = (96, 96)
            self.content_image_size = (96, 96)
            self.content_encoder_downsample_size = 3
            self.channel_attn = True
            self.content_start_channel = 64
            self.style_start_channel = 64
            self.model_type = "noise"
            self.guidance_type = "classifier-free"
            self.guidance_scale = 9.0 # Increased guidance scale for better character shape
            self.resolution = 96
            # Add other potential missing args that build_xx might need
            self.phase_2 = True
            self.sc_coefficient = 0.01
            # Scheduler args
            self.beta_scheduler = "scaled_linear"
            self.beta_start = 0.0001
            self.beta_end = 0.02
            self.num_train_timesteps = 1000
            self.inference_time_step = 20 # For DPM
            self.prediction_type = "epsilon"
            self.algorithm_type = "dpmsolver++" # Add this missing arg
    args = Args()
    print(f"DEBUG: Args attributes: {dir(args)}")
    if hasattr(args, 'beta_scheduler'):
        print(f"DEBUG: beta_scheduler = {args.beta_scheduler}")
    else:
        print("DEBUG: beta_scheduler is MISSING!")
    
    # Paths
    # style_dir is the dir_name (e.g. "My_Style"), we need full path
    
    # Special handling for "Current_Session"
    if style_dir == "Current_Session":
        # Dynamic lookup of latest checkpoint
        base_path = get_latest_ckpt("outputs/fine_tuning")
        if not base_path:
             # Fallback to hardcoded if lookup fails (shouldn't happen if trained)
             base_path = "outputs/fine_tuning/global_step_1500"
    else:
        # Check if style_dir is a full path (legacy) or just a name
        if os.sep in style_dir:
             base_path = style_dir
        else:
             base_path = os.path.join("outputs/styles", style_dir)
        
        if not os.path.exists(base_path):
            print(f"Warning: Style path {base_path} not found. Trying fallback logic...")
            # Fallback 1: Maybe it's in outputs/styles but passed as full path?
            # Fallback 2: Maybe it's a legacy fine-tuning path?
            
            # If the user selected a style that physically doesn't exist, we might crash.
            # But let's see if we can default to Current Session for safety?
            # No, that's confusing. Better to error out or try to find it.
            
            # Let's just try to look in outputs/styles again with just basename
            base_name = os.path.basename(style_dir)
            alt_path = os.path.join("outputs/styles", base_name)
            if os.path.exists(alt_path):
                base_path = alt_path
            else:
                 # Last resort: default to fine_tuning
                 print("Style definitely not found. Fallback to fine_tuning.")
                 base_path = get_latest_ckpt("outputs/fine_tuning") or "outputs/fine_tuning/global_step_1500"
    
    print(f"Final Model Path: {base_path}")
    
    # Load state dicts
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(os.path.join(base_path, "unet.pth"), map_location=device))
    
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(os.path.join(base_path, "style_encoder.pth"), map_location=device))
    
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(os.path.join(base_path, "content_encoder.pth"), map_location=device))
    
    model = FontDiffuserModelDPM(unet=unet, style_encoder=style_encoder, content_encoder=content_encoder)
    model.to(device)
    
    train_scheduler = build_ddpm_scheduler(args=args)
    
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    
    MODEL_CACHE[style_dir] = pipe
    return pipe

@app.post("/api/generate")
async def generate_text(request: GenerateRequest):
    try:
        pipe = load_model(request.style_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        import uuid
        
        # Prepare content font
        ttf_path = "web_ui/static/fonts/AlibabaPuHuiTi-3-55-Regular.ttf"
        # Fallback if not found
        if not os.path.exists(ttf_path):
             # Try searching
             for root, dirs, files in os.walk("web_ui/static/fonts"):
                 for file in files:
                     if file.endswith(".ttf"):
                         ttf_path = os.path.join(root, file)
                         break
        
        from utils import load_ttf
        font = load_ttf(ttf_path=ttf_path, fsize=96) # Match resolution
        
        # Prepare style image (Random from processed OR saved references)
        style_image_path = None
        
        # Logic Update: Strictly separate Current Session vs Saved Style sources
        
        if request.style_dir == "Current_Session":
            # 1. Try PROCESSED_DIR (Active session)
            if os.path.exists(PROCESSED_DIR):
                processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".jpg") and "grid" not in f]
                if processed_files:
                    style_image_path = os.path.join(PROCESSED_DIR, random.choice(processed_files))
            
            # 2. Fallback to fine_tuning backup (Previous session)
            if not style_image_path:
                backup_ref = "outputs/fine_tuning/reference_images"
                if os.path.exists(backup_ref):
                     ref_files = [f for f in os.listdir(backup_ref) if f.endswith(".jpg") and "grid" not in f]
                     if ref_files:
                         style_image_path = os.path.join(backup_ref, random.choice(ref_files))
        else:
            # Saved Style: ONLY look in the style's directory
            # Do NOT look in PROCESSED_DIR to avoid cross-contamination
             if os.sep in request.style_dir:
                 style_base = request.style_dir
             else:
                 style_base = os.path.join("outputs/styles", request.style_dir)
             
             ref_dir = os.path.join(style_base, "reference_images")
             if os.path.exists(ref_dir):
                 ref_files = [f for f in os.listdir(ref_dir) if f.endswith(".jpg") and "grid" not in f]
                 if ref_files:
                     style_image_path = os.path.join(ref_dir, random.choice(ref_files))
        
        # 3. Last Resort: Try data/MyFirstScript (for testing/fallback)
        if not style_image_path:
             fallback_dir = "data/MyFirstScript"
             if os.path.exists(fallback_dir):
                 ref_files = [f for f in os.listdir(fallback_dir) if f.endswith(".jpg") and "grid" not in f]
                 if ref_files:
                     style_image_path = os.path.join(fallback_dir, random.choice(ref_files))
                     print("Warning: Using fallback images from data/MyFirstScript")

        if not style_image_path:
            return JSONResponse(content={"message": "No style reference images found. Please upload a template or use a saved style with references."}, status_code=500)
            
        style_image_pil = Image.open(style_image_path).convert("RGB")
        
        # Preprocess style image: remove grid lines by cropping center and resizing back
        # Also maybe invert if needed? No, training data was likely grid-warped.
        # But we can try to crop 5% from edges to remove grid lines
        w, h = style_image_pil.size
        crop_margin = int(w * 0.08) # Crop 8% from edges
        style_image_pil = style_image_pil.crop((crop_margin, crop_margin, w - crop_margin, h - crop_margin))
        style_image_pil = style_image_pil.resize((96, 96), Image.BICUBIC)

        # Variation Mapping (Strong Perceptual Separation)
        variation_ratio = max(0.0, min(1.0, float(request.variation or 0.0)))
        variation_seed_bucket = int(variation_ratio * 1000)

        # Base seed includes chunk index so progressive generation doesn't repeat too similarly.
        base_seed = 1337 + (request.chunk_index * 100003) + (variation_seed_bucket * 97)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(base_seed)

        # Wider range than before to create stronger visible differences.
        pipe.guidance_scale = 6.0 + (variation_ratio * 7.0)
        num_inference_step = 14 + int(variation_ratio * 14)

        # Style perturbation strength for visible but controlled diversity.
        style_noise_strength = 0.01 + (variation_ratio * 0.08)
        
        # Transforms
        content_transform = transforms.Compose([
            transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        style_transform = transforms.Compose([
            transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        style_tensor_base = style_transform(style_image_pil).unsqueeze(0).to(device)
        
        generated_images = []
        
        # Generate each char
        # Limit length for demo performance
        text = request.text[:20] 
        
        for char_idx, char in enumerate(text):
            if not char.strip():
                continue
                
            # Content Image
            try:
                content_pil = ttf2im(font, char, fsize=96)
                if content_pil is None:
                    # Fallback for missing glyphs or spaces
                    continue
            except:
                continue

            # Character-level deterministic seed for stronger, controllable variation.
            char_seed = base_seed + ((char_idx + 1) * 7919) + ord(char[0])
            torch.manual_seed(char_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(char_seed)

            # Light geometric jitter on content glyph, grows with variation.
            if variation_ratio > 0:
                max_angle = 1.0 + variation_ratio * 7.0
                angle = (random.random() - 0.5) * 2.0 * max_angle
                content_pil = content_pil.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

            style_tensor = style_tensor_base
            if variation_ratio > 0:
                style_noise = torch.randn_like(style_tensor_base) * style_noise_strength
                style_tensor = torch.clamp(style_tensor_base + style_noise, -1.0, 1.0)
                
            content_tensor = content_transform(content_pil).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                images = pipe.generate(
                    content_images=content_tensor,
                    style_images=style_tensor,
                    batch_size=1,
                    order=2,
                    num_inference_step=num_inference_step,
                    content_encoder_downsample_size=3,
                    t_start=None,
                    t_end=None,
                    dm_size=(96, 96),
                    algorithm_type="dpmsolver++",
                    skip_type="time_uniform",
                    method="multistep",
                    correcting_x0_fn=None
                )
                generated_images.append(images[0])
        
        # Save results
        output_urls = []
        gen_dir = "web_ui/static/generated"
        os.makedirs(gen_dir, exist_ok=True)
        
        timestamp_ms = int(time.time() * 1000)
        for i, img in enumerate(generated_images):
            filename = f"gen_{timestamp_ms}_{request.chunk_index}_{i}_{uuid.uuid4().hex[:8]}.png"
            path = os.path.join(gen_dir, filename)
            
            # Post-processing: Convert to Transparent Ink
            # Strategy:
            # 1. The model generates Black Ink on White Paper.
            # 2. We want to extract the Ink.
            # 3. User requested: Invert (White Ink on Black) -> Make Black Transparent.
            
            from PIL import ImageOps
            
            # Ensure we are working with the raw generated image
            img = img.convert("RGB")
            
            # 1. Invert: Black Ink (0) becomes White Ink (255). White Paper (255) becomes Black Paper (0).
            img_inverted = ImageOps.invert(img)
            
            # 2. Create Alpha Mask from the inverted image (Grayscale)
            # Brighter pixels (Ink) -> More Opaque. Darker pixels (Paper) -> More Transparent.
            mask = img_inverted.convert("L")
            
            # 3. Create a solid White image
            white_canvas = Image.new("RGBA", img.size, (255, 255, 255, 0))
            
            # 4. Apply the mask to the alpha channel of the white canvas
            # Result: White Ink on Transparent Background
            white_canvas.putalpha(mask)
            
            # Save
            white_canvas.save(path, format="PNG")
            output_urls.append(f"/static/generated/{filename}")
            
        return JSONResponse(content={"images": output_urls})
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        # Log to file for debugging
        with open("server_error.log", "w") as f:
            f.write(error_msg)
        return JSONResponse(content={"message": f"Server Error: {str(e)} \nTraceback: {error_msg}"}, status_code=500)

@app.get("/api/training_status")
async def get_training_status():
    global TRAINING_STATUS
    
    # Disable "Smart Recovery" to prevent false positives during retraining debugging
    # if TRAINING_STATUS["state"] == "idle":
    #    final_ckpt = os.path.join("outputs/fine_tuning", "global_step_2000", "total_model.pth")
    #    if os.path.exists(final_ckpt):
    #        TRAINING_STATUS["state"] = "completed"
    #        TRAINING_STATUS["progress"] = 100
    #        TRAINING_STATUS["log"].append("Recovered previous training session.")
            
    # Return last 50 logs to keep it light
    return JSONResponse(content={
        "state": TRAINING_STATUS["state"],
        "progress": TRAINING_STATUS["progress"],
        "log": TRAINING_STATUS["log"][-50:],
        "error": TRAINING_STATUS["error"]
    })

@app.get("/api/characters")
async def get_characters():
    global PROCESSED_CHARS
    response_data = []
    
    # Sort just in case
    # PROCESSED_CHARS might be empty initially
    
    for item in PROCESSED_CHARS:
        filename = os.path.basename(item["path"])
        # Extract index from filename "0_永.jpg"
        try:
            idx = int(filename.split('_')[0])
        except:
            idx = 0
            
        response_data.append({
            "index": idx,
            "char": item["char"],
            "url": f"/sliced_images/{filename}",
            "confidence": item["confidence"],
            "bbox": item.get("bbox", [0,0,100,100])
        })
        
    return response_data

class FontNameRequest(BaseModel):
    name: str

@app.post("/api/save_font_name")
async def save_font_name(request: FontNameRequest):
    try:
        # Create a safe directory name from the font name
        safe_name = "".join([c for c in request.name if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not safe_name:
            safe_name = "My_Custom_Font"
        
        # Define styles directory
        styles_dir = "outputs/styles"
        target_dir = os.path.join(styles_dir, safe_name)
        
        # Source directory (current training output)
        source_dir = get_latest_ckpt("outputs/fine_tuning")
        
        if not source_dir or not os.path.exists(source_dir):
            return JSONResponse(content={"message": "Training output not found. Please train first."}, status_code=404)
            
        # Copy files
        import shutil
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy model files
        for filename in ["total_model.pth", "unet.pth", "style_encoder.pth", "content_encoder.pth"]:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(target_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Save reference images (Crucial for generation later!)
        ref_dir = os.path.join(target_dir, "reference_images")
        os.makedirs(ref_dir, exist_ok=True)
        
        # Copy from PROCESSED_DIR
        processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".jpg") and "grid" not in f]
        
        # If PROCESSED_DIR is empty (e.g. session cleared), try to copy from fine_tuning backup
        if not processed_files:
            backup_ref = os.path.join(source_dir, "reference_images")
            if os.path.exists(backup_ref):
                processed_files = [f for f in os.listdir(backup_ref) if f.endswith(".jpg") and "grid" not in f]
                # Update source path for copy loop
                source_root = backup_ref
            else:
                source_root = PROCESSED_DIR # Will fail/empty
        else:
            source_root = PROCESSED_DIR

        # Limit to 20 random images to save space but provide variety
        import random
        if len(processed_files) > 20:
            selected_files = random.sample(processed_files, 20)
        else:
            selected_files = processed_files
            
        for fname in selected_files:
            src = os.path.join(source_root, fname)
            dst = os.path.join(ref_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            
        # Also save the grid image as backup context
        grid_src = os.path.join(source_root, "grid_warped.jpg")
        if os.path.exists(grid_src):
            shutil.copy2(grid_src, os.path.join(ref_dir, "grid_warped.jpg"))
        
        # Save metadata
        import json
        # Extract step number from source_dir
        try:
            step_name = os.path.basename(source_dir)
        except:
            step_name = "global_step_unknown"
            
        metadata = {
            "font_name": request.name,
            "dir_name": safe_name,
            "created_at": time.time(),
            "base_model": step_name
        }
        
        with open(os.path.join(target_dir, "style_config.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        # Also save to legacy config for backward compatibility if needed, 
        # but primarily we rely on the styles folder now.
        config_path = "outputs/fine_tuning/font_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        return JSONResponse(content={"message": "Font style saved successfully", "path": target_dir})
    except Exception as e:
        print(f"Error saving font: {e}")
        return JSONResponse(content={"message": str(e)}, status_code=500)

@app.get("/api/styles")
async def get_styles():
    styles_dir = "outputs/styles"
    styles = []
    import json
    
    # 1. Add "Last Training Session" if exists AND NOT SAVED YET
    # We check if outputs/fine_tuning/font_config.json exists. If so, it was saved.
    is_saved = os.path.exists("outputs/fine_tuning/font_config.json")
    
    last_session_path = get_latest_ckpt("outputs/fine_tuning")
    
    if last_session_path and os.path.exists(last_session_path) and not is_saved:
        # Check if valid model files exist
        if os.path.exists(os.path.join(last_session_path, "unet.pth")):
            styles.append({
                "font_name": "Current Training Session",
                "dir_name": "Current_Session", # Special keyword
                "created_at": os.stat(last_session_path).st_mtime
            })
    
    if os.path.exists(styles_dir):
        # Iterate over subdirectories
        for entry in os.scandir(styles_dir):
            if entry.is_dir():
                # Skip legacy MyFirstScript if it's broken or not needed, 
                # OR check if it has valid model files.
                
                # Check for model existence
                if not os.path.exists(os.path.join(entry.path, "unet.pth")):
                    continue

                config_path = os.path.join(entry.path, "style_config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            styles.append(data)
                    except:
                        pass
                else:
                    # Fallback if no config
                    styles.append({
                        "font_name": entry.name,
                        "dir_name": entry.name,
                        "created_at": entry.stat().st_ctime
                    })
    
    # Sort by creation time (newest first)
    styles.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return styles

@app.post("/api/delete_style")
async def delete_style(request: FontNameRequest):
    # Here request.name is actually the dir_name or we find by font_name
    # Let's assume frontend sends dir_name for precision, or we search.
    # For simplicity, let's search by font_name or dir_name match.
    
    styles_dir = "outputs/styles"
    target_dir = None
    
    # Try exact dir match first
    potential_dir = os.path.join(styles_dir, request.name)
    if os.path.exists(potential_dir):
        target_dir = potential_dir
    else:
        # Search by font_name in jsons
        for entry in os.scandir(styles_dir):
            if entry.is_dir():
                config_path = os.path.join(entry.path, "style_config.json")
                if os.path.exists(config_path):
                    import json
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if data.get("font_name") == request.name:
                                target_dir = entry.path
                                break
                    except:
                        pass
    
    if target_dir and os.path.exists(target_dir):
        import shutil
        try:
            shutil.rmtree(target_dir)
            return JSONResponse(content={"message": "Style deleted"})
        except Exception as e:
            return JSONResponse(content={"message": str(e)}, status_code=500)
    
    return JSONResponse(content={"message": "Style not found"}, status_code=404)

@app.get("/api/font_config")
async def get_font_config():
    config_path = "outputs/fine_tuning/font_config.json"
    if os.path.exists(config_path):
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(content={})

@app.post("/api/update_slice")
async def update_slice(request: CropRequest):
    global PROCESSED_CHARS
    
    grid_path = os.path.join(PROCESSED_DIR, "grid_warped.jpg")
    if not os.path.exists(grid_path):
        return JSONResponse(content={"message": "Grid image not found"}, status_code=404)
        
    # Perform crop
    result = crop_and_save(grid_path, request.dict(), PROCESSED_DIR)
    
    if result:
        # Update global store
        for item in PROCESSED_CHARS:
            # Check by filename match logic or index
            # Path ends with {index}_{char}.jpg
            fname = os.path.basename(item["path"])
            if fname.startswith(f"{request.index}_"):
                item["bbox"] = result["bbox"]
                # Path is same (overwritten)
                break
        return JSONResponse(content={"message": "Updated successfully", "url": f"/sliced_images/{os.path.basename(result['path'])}?t={os.urandom(4).hex()}"})
    else:
        return JSONResponse(content={"message": "Crop failed"}, status_code=500)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global PROCESSED_CHARS
    # Clear previous results to prevent stale data
    PROCESSED_CHARS = []
    
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Trigger slicing immediately
        print(f"Processing uploaded file: {file_path}")
        results = process_scanned_template(file_path, PROCESSED_DIR)
        
        if results:
            PROCESSED_CHARS = results
            return JSONResponse(content={
                "filename": file.filename, 
                "message": f"Processed {len(results)} characters successfully",
                "count": len(results)
            })
        else:
            return JSONResponse(content={
                "filename": file.filename,
                "message": "Upload successful but failed to detect grid/markers",
                "warning": True
            })
            
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"message": str(e)}, status_code=500)

if __name__ == "__main__":
    print("Starting WebUI on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
