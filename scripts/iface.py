import gradio as gr
from modules import script_callbacks, shared_items, sd_vae, sd_models, images
from modules.paths_internal import default_output_dir

from diffusers.models import AutoencoderKL
from diffusers import EulerDiscreteScheduler
from torchvision import transforms
from PIL import Image
import random

import os
import gc
import torch
from demofusion.pipeline_demofusion_sdxl import DemoFusionSDXLStableDiffusionPipeline
from demofusion.pipeline_demofusion_sd import DemoFusionSDStableDiffusionPipeline

df_out = os.path.join(default_output_dir, 'demofusion')

# img2img-part
def load_and_process_image(pil_image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(pil_image)
    image = image.unsqueeze(0).half()
    return image

def set_checkpoint_model(selected_model):
    global model_ckpt
    found_model = sd_models.get_closet_checkpoint_match(selected_model)
    if found_model:
        model_ckpt = found_model.filename

def set_vae_model(selected_vae):
    global model_vae
    model_vae = sd_vae.vae_dict.get(selected_vae, selected_vae)

def set_lora_model(selected_lora):
    global model_lora
    model_lora = selected_lora

def set_base_model(selected_base):
    global model_base
    model_base = selected_base

def generate_images(prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, cosine_scale_1,
                    cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, set_lora_scale, input_image,
                    cb_multidecoder, clip_skip, scale_num):
    if input_image:
        image_lr = load_and_process_image(input_image).to('cuda')
    else:
        image_lr = None
    
    if model_base == "SD1.5":
        pipebase = DemoFusionSDStableDiffusionPipeline
        stride = stride / 2
    else:
        pipebase = DemoFusionSDXLStableDiffusionPipeline

    if model_vae == "Not used":
        pipe = pipebase.from_single_file(model_ckpt, use_safetensors=True,
                                                                      torch_dtype=torch.float16,
                                                                      low_cpu_mem_usage=False,
                                                                      ignore_mismatched_sizes=True)
    else:
        vae = AutoencoderKL.from_single_file(model_vae, torch_dtype=torch.float16)
        pipe = pipebase.from_single_file(model_ckpt, vae=vae, use_safetensors=True,
                                                                      torch_dtype=torch.float16,
                                                                      low_cpu_mem_usage=False,
                                                                      ignore_mismatched_sizes=True)

    if model_lora == "Not used":
        pass
    else:
        pipe.load_lora_weights(model_lora)
        pipe.fuse_lora(lora_scale=set_lora_scale)

    pipe = pipe.to("cuda")
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))

    df_images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                  height=int(height), width=int(width), view_batch_size=int(view_batch_size), stride=int(stride),
                  num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                  cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3,
                  sigma=sigma,
                  multi_decoder=bool(cb_multidecoder), show_image=False, image_lr=image_lr,
                  clip_skip=int(clip_skip), scale_num=int(scale_num)
                  )

    df_info = [
            f'Prompt: {prompt}',
            f'Negative prompt: {negative_prompt}',
            f'Width: {width}',
            f'Height: {height}',
            f'Steps: {num_inference_steps}',
            f'CFG Scale: {guidance_scale}',
            f'Seed: {int(seed)}',
            f'Clip skip: {int(clip_skip)}']

    df_info_ext = [
            f'Prompt: {prompt}',
            f'Negative prompt: {negative_prompt}',
            f'Width: {width}',
            f'Height: {height}',
            f'Steps: {num_inference_steps}',
            f'CFG Scale: {guidance_scale}',
            f'C1: {cosine_scale_1}',
            f'C2: {cosine_scale_2}',
            f'C3: {cosine_scale_3}',
            f'Sigma: {sigma}',
            f'View batch size: {view_batch_size}',
            f'Stride: {stride}',
            f'Seed: {int(seed)}',
            f'Clip skip: {int(clip_skip)}',
            f'Scale: x{scale_num}']

    images.save_image(df_images[0], df_out, 'df', int(seed), info=df_info, pnginfo_section_name='DemoFusion')
    images.save_image(df_images[-1], df_out, 'df', int(seed), info=df_info_ext, pnginfo_section_name='DemoFusion')
    pipe = None
    gc.collect()
    torch.cuda.empty_cache()

    return df_images[0], df_images[-1]


def on_ui_tabs():
    import networks
    networks.available_networks
    model_list = shared_items.list_checkpoint_tiles(False)
    lora_list = ['Not used'] + [lora.filename for lora in networks.available_networks.values()]
    vae_list = ['Not used'] + list(sd_vae.vae_dict)
    base_list = ['SDXL', 'SD1.5']
    with gr.Blocks(analytics_enabled=False) as DF_Blocks:
        with gr.Row():
            sd_ckpt_file = gr.Dropdown(model_list, label="Model", info="Stable Diffusion Model", scale=30)
            setattr(sd_ckpt_file,"do_not_save_to_config",True)
            sd_vae_file = gr.Dropdown(vae_list, label="VAE (optional)", info="Vae Model", scale=30)
            setattr(sd_vae_file,"do_not_save_to_config",True)
            sd_lora_file = gr.Dropdown(lora_list, label="LoRA (optional)", info="LoRA Model", scale=30)
            setattr(sd_lora_file,"do_not_save_to_config",True)
            set_lora_scale = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.85, label="Weight", info="Lora scale",
                                       scale=10)
        with gr.Row():
            with gr.Column(scale=40):
                with gr.Row():
                    base_model = gr.Dropdown(base_list, label="Base Model")
                    setattr(base_model,"do_not_save_to_config",True)
                    cb_multidecoder = gr.Checkbox(label="Multidecoder", value=True, info="Use multidecoder?")
                with gr.Accordion('img2img (SDXL works only with square images)', open=False):
                    m_image_input = gr.Image(type="pil", label="Input Image")
                m_prompt = gr.Textbox(label="Prompt")
                m_negative_prompt = gr.Textbox(label="Negative Prompt",
                                               value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
                with gr.Row():
                    with gr.Column(min_width=150):
                        m_width = gr.Slider(minimum=384, maximum=2048, step=8, value=1024, label="Width")
                        m_height = gr.Slider(minimum=384, maximum=2048, step=8, value=1024, label="Height")
                    with gr.Column(min_width=150):
                        m_num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Sampling Steps")
                        m_guidance_scale = gr.Slider(minimum=1.1, maximum=20, step=0.1, value=7.5, label="CFG Scale")
                with gr.Row():
                    scale_num = gr.Slider(minimum=1, maximum=8, step=1, value=1, label="Scale Factor")
                with gr.Row():
                    m_seed = gr.Number(scale=10, label="Seed", value=2013)
                    clip_skip = gr.Number(scale=1, min_width=85, label="Clip skip", minimum=1, maximum=12, step=1, value=2)
                with gr.Accordion('Additional parameters', open=False):
                    m_sigma = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Sigma")
                    with gr.Row():
                        m_cosine_scale_1 = gr.Slider(minimum=0, maximum=5, step=0.1, value=3, label="Cosine Scale 1")
                        m_cosine_scale_2 = gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 2")
                        m_cosine_scale_3 = gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 3")
                    with gr.Row():
                        m_stride = gr.Slider(minimum=8, maximum=96, step=8, value=64, label="Stride")
                        m_view_batch_size = gr.Slider(minimum=4, maximum=32, step=4, value=16, label="View Batch Size")

            with gr.Column(scale=60):
                with gr.Row():
                    submit_btn = gr.Button(value="Generate", variant="primary")
                    submit_random_btn = gr.Button(value="Random seed", variant="secondary")
                    # cancel_btn = gr.Button(value="Cancel")
                with gr.Row():
                    main_outputs = gr.Gallery(label="Generated Images")
                    # outputs=ImageSlider(label="Comparison of SDXL and DemoFusion")

        DF_Blocks.load(
            lambda: [gr.update(value=model_list[0]), gr.update(value=vae_list[0]), gr.update(value=lora_list[0]), gr.update(value=base_list[0])],
            None,
            [sd_ckpt_file, sd_vae_file, sd_lora_file, base_model])
        main_inputs = [m_prompt, m_negative_prompt, m_width, m_height, m_num_inference_steps,
                       m_guidance_scale, m_cosine_scale_1, m_cosine_scale_2, m_cosine_scale_3,
                       m_sigma, m_view_batch_size, m_stride, m_seed, set_lora_scale, m_image_input, cb_multidecoder,
                       clip_skip, scale_num]
        sd_ckpt_file.change(set_checkpoint_model, inputs=sd_ckpt_file, outputs=sd_ckpt_file.value)
        sd_vae_file.change(set_vae_model, inputs=sd_vae_file, outputs=sd_vae_file.value)
        sd_lora_file.change(set_lora_model, inputs=sd_lora_file, outputs=sd_lora_file.value)
        base_model.change(set_base_model, inputs=base_model, outputs=base_model.value)
        submit_btn.click(generate_images, inputs=main_inputs, outputs=main_outputs)
        submit_random_btn.click(lambda: gr.update(value=random.randrange(1, 4294967295)), None, m_seed).then(
            generate_images, inputs=main_inputs, outputs=main_outputs)
        # cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event])
    return [(DF_Blocks, "DemoFusion", "DemoFusion")]


script_callbacks.on_ui_tabs(on_ui_tabs)
