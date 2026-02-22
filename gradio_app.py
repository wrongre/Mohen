import random
import gradio as gr
from sample import (arg_parse, 
                    sampling,
                    load_fontdiffuer_pipeline)


def run_fontdiffuer(source_image, 
                    character, 
                    reference_image,
                    sampling_step,
                    guidance_scale,
                    batch_size):
    args.character_input = False if source_image is not None else True
    args.content_character = character
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    args.seed = random.randint(0, 10000)
    out_image = sampling(
        args=args,
        pipe=pipe,
        content_image=source_image,
        style_image=reference_image)
    return out_image


if __name__ == '__main__':
    args = arg_parse()
    args.demo = True
    args.ckpt_dir = 'ckpt'
    
    import torch
    import os
    # Auto-detect device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {args.device}")
    
    # Auto-detect font
    if os.path.exists('C:/Windows/Fonts/simsun.ttc'):
        args.ttf_path = 'C:/Windows/Fonts/simsun.ttc'
    else:
        args.ttf_path = 'ttf/KaiXinSongA.ttf'
    print(f"Using font: {args.ttf_path}")

    # load fontdiffuer pipeline
    pipe = load_fontdiffuer_pipeline(args=args)

    with gr.Blocks(title="Clonify-Ink | AI Font Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🖌️ Clonify-Ink: Your Personal AI Font Designer
        Create your own digital font from just a few handwritten characters.
        """)
        
        with gr.Tabs():
            # Tab 1: Train/Clone
            with gr.TabItem("🎓 1. Clone Your Style"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Step 1: Upload Template")
                        gr.Markdown("Please download our [Standard Template PDF](#) and write the characters in the boxes.")
                        template_upload = gr.File(label="Upload Scanned Template (PDF/Image)", file_types=[".pdf", ".png", ".jpg"])
                        
                        gr.Markdown("### Step 2: Verify Slicing")
                        # Mocking grid gallery
                        slice_gallery = gr.Gallery(label="Detected Characters", show_label=True, elem_id="slice_gallery", columns=6, rows=5, height=400)
                        
                        verify_btn = gr.Button("✅ Confirm & Start Training", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Training Status")
                        progress_bar = gr.Label(value="Ready to train...", label="Progress")
                        logs = gr.Textbox(label="System Logs", lines=10, interactive=False)
            
            # Tab 2: Generate
            with gr.TabItem("✨ 2. Generate Font"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_select = gr.Dropdown(choices=["MyStyle_v1", "MyStyle_v2"], label="Select Your Style Model", value="MyStyle_v1")
                        text_input = gr.Textbox(label="Enter Text to Generate", placeholder="Type any Chinese characters here...", lines=3)
                        generate_btn = gr.Button("🎨 Generate", variant="primary")
                        
                    with gr.Column(scale=1):
                        output_gallery = gr.Gallery(label="Generated Results", columns=4)
                        download_btn = gr.Button("⬇️ Download as Image Pack")

    demo.launch(server_name="127.0.0.1", server_port=7861, debug=True)
