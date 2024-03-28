# DemoFusion interface inside stable-diffusion-webui
Readme will be updated soon!

[Original project](https://ruoyidu.github.io/demofusion/demofusion.html) 

[Original project GitHub](https://github.com/PRIS-CV/DemoFusion)

- Added support for stable diffusion files
- Added support for custom HuggingFace models
- Rebuild UI
- something else?

## Installation:
Extensions => Install from URL => `https://github.com/sebaxakerhtc/sd-webui-demofusion.git` => Install
Then switch to installed tab, click Apply and restart UI

## Usage:
Just input a double (triple, etc.) size of the image you want to generate and wait for your image.
Image size is hardcoded for now - so, your images larges width or height will be setted to 1024, 2048, etc
For example I want to generate an image with original size = 832 x 1216 px. I set width to 1664 and height to 2432.
The output will be 1401 x 2048, because the largest width/height must be divisible by 1024.
But if you will set image size 1401 x 2048 - the original image size of generated image will be 704 x 1024 - it's
bad resolution for SDXL.

![изображение](https://github.com/sebaxakerhtc/sd-webui-demofusion/assets/32651506/c3af3c42-0609-4f6f-96cd-e1c8f095940b)

To do:
- img2img
- LoRA support
- ControlNet, maybe...
- Custom resolutions
- SD 1.5 adaptation? Maybe...
