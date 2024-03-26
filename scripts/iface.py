import gradio as gr
from pathlib import Path
from modules import script_callbacks, shared
import json
import os
import torch

from demofusion.pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
# from gradio_imageslider import ImageSlider

# Webui root path
ROOT_DIR = Path().absolute()

def generate_images(prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed):
    model_ckpt = "camenduru/DemoFusion"
    pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))

    images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                  height=int(height), width=int(width), view_batch_size=int(view_batch_size), stride=int(stride),
                  num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                  cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3, sigma=sigma,
                  multi_decoder=True, show_image=False
                 )

    return (images[0], images[-1])

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as GPT_Blocks:
        gr.HTML(
            "<p id='demofusion',style=\"margin-bottom:0.75em\">DemoFusion extension for Automatic1111</p>")
        with gr.Row():
            with gr.Column(scale=45):
                inputs = [
                    gr.Textbox(label="Prompt"),
                    gr.Textbox(label="Negative Prompt", value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic"),
                    gr.Slider(minimum=1024, maximum=4096, step=8, value=2048, label="Height"),
                    gr.Slider(minimum=1024, maximum=4096, step=8, value=2048, label="Width"),
                    gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Num Inference Steps"),
                    gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale"),
                    gr.Slider(minimum=0, maximum=5, step=0.1, value=3, label="Cosine Scale 1"),
                    gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 2"),
                    gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 3"),
                    gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Sigma"),
                    gr.Slider(minimum=4, maximum=32, step=4, value=16, label="View Batch Size"),
                    gr.Slider(minimum=8, maximum=96, step=8, value=64, label="Stride"),
                    gr.Number(label="Seed", value=2013)
                ]
                submit_btn = gr.Button("Generate")
            with gr.Column(scale=80):
                outputs=gr.Gallery(label="Generated Images")
                # outputs=ImageSlider(label="Comparison of SDXL and DemoFusion")
        submit_btn.click(generate_images, inputs, outputs)

    return [(GPT_Blocks, "DemoFusion", "DemoFusion")]


script_callbacks.on_ui_tabs(on_ui_tabs)