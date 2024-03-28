import gradio as gr
from pathlib import Path
from modules import script_callbacks, shared, paths, modelloader
from diffusers.models import AutoencoderKL
import json
import os
import torch
import os.path

from demofusion.pipeline_demofusion_sdxl import DemoFusionSDXLStableDiffusionPipeline
# from gradio_imageslider import ImageSlider

# Webui root path
ROOT_DIR = Path().absolute()
model_dir = "Stable-diffusion"
vae_dir = "VAE"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
vae_path = os.path.abspath(os.path.join(paths.models_path, vae_dir))
model_list = modelloader.load_models(model_path=model_path, ext_filter=[".ckpt", ".safetensors"])
vae_list = modelloader.load_models(model_path=vae_path, ext_filter=[".ckpt", ".safetensors"])
sd_model_path = DemoFusionSDXLStableDiffusionPipeline.from_single_file
sd_vae_path = AutoencoderKL.from_single_file
hf_model_path = DemoFusionSDXLStableDiffusionPipeline.from_pretrained
hf_vae_path = AutoencoderKL.from_pretrained
df_model_path = sd_model_path
df_vae_path = sd_vae_path

def set_checkpoint_model(selected_model):
    global model_ckpt
    model_ckpt = selected_model
    
def set_vae_model(selected_vae):
    global model_vae
    model_vae = selected_vae
    
def set_path_sd():
    global df_model_path, df_vae_path
    df_model_path = sd_model_path
    df_vae_path = sd_vae_path
    
def set_path_hf():
    global df_model_path, df_vae_path, model_ckpt, model_vae
    df_model_path = hf_model_path
    df_vae_path = hf_vae_path
    model_ckpt = "camenduru/DemoFusion"
    # model_vae = "madebyollin/sdxl-vae-fp16-fix"

def generate_images(prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed):
    # vae = df_vae_path(model_vae)
    pipe = df_model_path(model_ckpt, use_safetensors=True, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))

    images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                  height=int(height), width=int(width), view_batch_size=int(view_batch_size), stride=int(stride),
                  num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                  cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3, sigma=sigma,
                  multi_decoder=True, show_image=True
                 )

    return (images[0], images[-1])

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as DF_Blocks:
        with gr.Row():
            with gr.Tab("Stable Diffusion") as sd:
                with gr.Row():
                    sd_ckpt_file = gr.Dropdown(model_list, label="Model (Only SDXL Models are supported for now)", info="Stable Diffusion Model")
                    sd_vae_file = gr.Dropdown(vae_list, label="VAE", info="Vae Model (coming soon)")
            with gr.Tab("HuggingFace") as hf:
                with gr.Row():
                    hf_ckpt_file = gr.Textbox(label="Model (Only SDXL Models are supported for now)", placeholder="default:camenduru/DemoFusion", info="HuggingFace Model Path")
                    hf_vae_file = gr.Textbox(label="VAE", placeholder="disabled because of errors", interactive=False, info="VAE Path")
        with gr.Row():
            with gr.Column(scale=45):
                inputs = [
                    gr.Textbox(label="Prompt"),
                    gr.Textbox(label="Negative Prompt", value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic"),
                    gr.Slider(minimum=1024, maximum=4096, step=8, value=2048, label="Width"),
                    gr.Slider(minimum=1024, maximum=4096, step=8, value=2048, label="Height"),
                    gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Num Inference Steps"),
                    gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale"),
                    gr.Slider(minimum=0, maximum=5, step=0.1, value=3, label="Cosine Scale 1"),
                    gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 2"),
                    gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 3"),
                    gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Sigma"),
                    gr.Slider(minimum=4, maximum=32, step=4, value=16, label="View Batch Size"),
                    gr.Slider(minimum=8, maximum=96, step=8, value=64, label="Stride"),
                    gr.Number(label="Seed", value=2013)
                ]
            with gr.Column(scale=80):
                with gr.Row():
                    submit_btn = gr.Button(value="Generate", variant="primary")
                    # cancel_btn = gr.Button(value="Cancel")
                with gr.Row():
                    outputs=gr.Gallery(label="Generated Images")
                    # outputs=ImageSlider(label="Comparison of SDXL and DemoFusion")

        sd_ckpt_file.change(set_checkpoint_model, inputs=sd_ckpt_file, outputs=sd_ckpt_file.value)
        sd_vae_file.change(set_vae_model, inputs=sd_vae_file, outputs=sd_vae_file.value)
        hf_ckpt_file.change(set_checkpoint_model, inputs=hf_ckpt_file, outputs=None)
        hf_vae_file.change(set_vae_model, inputs=hf_vae_file, outputs=None)
        sd.select(set_path_sd)
        hf.select(set_path_hf)
        submit_btn.click(generate_images, inputs, outputs)
        # click_event = submit_btn.click(generate_images, inputs, outputs)
        # cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event])

    return [(DF_Blocks, "DemoFusion", "DemoFusion")]


script_callbacks.on_ui_tabs(on_ui_tabs)
