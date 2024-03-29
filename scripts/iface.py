import gradio as gr
from pathlib import Path
from modules import script_callbacks, shared, paths, modelloader
from diffusers.models import AutoencoderKL
import json
import os
import torch
import os.path
import random

from demofusion.pipeline_demofusion_sdxl import DemoFusionSDXLStableDiffusionPipeline
# from gradio_imageslider import ImageSlider

# Webui root path
ROOT_DIR = Path().absolute()
model_dir = "Stable-diffusion"
vae_dir = "VAE"
lora_dir = "Lora"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
vae_path = os.path.abspath(os.path.join(paths.models_path, vae_dir))
lora_path = os.path.abspath(os.path.join(paths.models_path, lora_dir))
model_list = modelloader.load_models(model_path=model_path, ext_filter=[".ckpt", ".safetensors"])
vae_list = modelloader.load_models(model_path=vae_path, ext_filter=[".ckpt", ".safetensors"])
vae_list.append("Not used")
lora_list = modelloader.load_models(model_path=lora_path, ext_filter=[".safetensors"])
lora_list.append("Not used")

def set_checkpoint_model(selected_model):
    global model_ckpt
    model_ckpt = selected_model
    
def set_vae_model(selected_vae):
    global model_vae
    model_vae = selected_vae
    
def set_lora_model(selected_lora):
    global model_lora
    model_lora = selected_lora

def generate_images(prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, set_lora_scale):
    
    if model_vae == "Not used":
        pipe = DemoFusionSDXLStableDiffusionPipeline.from_single_file(model_ckpt, use_safetensors=True, torch_dtype=torch.float16, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    else:
        vae = AutoencoderKL.from_single_file(model_vae)
        pipe = DemoFusionSDXLStableDiffusionPipeline.from_single_file(model_ckpt, vae=vae, use_safetensors=True, torch_dtype=torch.float16, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

    if model_lora == "Not used":
        pass
    else:
        pipe.load_lora_weights(model_lora)
        pipe.fuse_lora(lora_scale = set_lora_scale)

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
            sd_ckpt_file = gr.Dropdown(sorted(model_list), label="Model (Only SDXL Models are supported for now)", info="Stable Diffusion Model", scale=30)
            sd_vae_file = gr.Dropdown(sorted(vae_list), label="VAE (optional)", info="Vae Model", scale=30)
            sd_lora_file = gr.Dropdown(sorted(lora_list), label="LoRA (optional)", info="LoRA Model", scale=30)
            set_lora_scale = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.85, label="Weight", info="Lora scale", scale=10)
        with gr.Row():
            with gr.Column(scale=45):
                m_prompt = gr.Textbox(label="Prompt")
                m_negative_prompt = gr.Textbox(label="Negative Prompt", value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
                m_width = gr.Slider(minimum=768, maximum=8192, step=8, value=2048, label="Width")
                m_height = gr.Slider(minimum=768, maximum=8192, step=8, value=2048, label="Height")
                m_num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Num Inference Steps")
                m_guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale")
                m_cosine_scale_1 = gr.Slider(minimum=0, maximum=5, step=0.1, value=3, label="Cosine Scale 1")
                m_cosine_scale_2 = gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 2")
                m_cosine_scale_3 = gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 3")
                m_sigma = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Sigma")
                m_view_batch_size = gr.Slider(minimum=4, maximum=32, step=4, value=16, label="View Batch Size")
                m_stride = gr.Slider(minimum=8, maximum=96, step=8, value=64, label="Stride")
                m_seed = gr.Number(label="Seed", value=2013)
            with gr.Column(scale=80):
                with gr.Row():
                    submit_btn = gr.Button(value="Generate", variant="primary", scale=90)
                    # cancel_btn = gr.Button(value="Cancel")
                with gr.Row():
                    main_outputs=gr.Gallery(label="Generated Images")
                    # outputs=ImageSlider(label="Comparison of SDXL and DemoFusion")

        main_inputs = [m_prompt, m_negative_prompt, m_width, m_height, 
        m_num_inference_steps, m_guidance_scale, m_cosine_scale_1, 
        m_cosine_scale_2, m_cosine_scale_3, m_sigma, m_view_batch_size, m_stride, m_seed, set_lora_scale]
        sd_ckpt_file.change(set_checkpoint_model, inputs=sd_ckpt_file, outputs=sd_ckpt_file.value)
        sd_vae_file.change(set_vae_model, inputs=sd_vae_file, outputs=sd_vae_file.value)
        sd_lora_file.change(set_lora_model, inputs=sd_lora_file, outputs=sd_lora_file.value)
        submit_btn.click(generate_images, inputs=main_inputs, outputs=main_outputs)
        DF_Blocks.load(lambda: [gr.update(value=model_list[0]), gr.update(value="Not used"), gr.update(value="Not used"), gr.update(value=random.randrange(1, 4294967295))], 
        None, 
        [sd_ckpt_file, sd_vae_file, sd_lora_file, m_seed])
        # cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event])

    return [(DF_Blocks, "DemoFusion", "DemoFusion")]


script_callbacks.on_ui_tabs(on_ui_tabs)
