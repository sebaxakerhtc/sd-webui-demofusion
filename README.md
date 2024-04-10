# DemoFusion interface inside stable-diffusion-webui

![изображение](https://github.com/sebaxakerhtc/sd-webui-demofusion/assets/32651506/801c9eee-1d37-40b3-83fe-509562c5c9fc)

[Original project](https://ruoyidu.github.io/demofusion/demofusion.html) 

[Original project GitHub](https://github.com/PRIS-CV/DemoFusion)

<details>
<summary><b>Changelog</b></summary>

10.04.2024
- added custom resolutions
- rebuild UI
- removed lowvram (Doesn't work with custom resolutions. Only square images works)
- SD1.5 support is on it's way. All works fine - hard testings. Wait for it!
- maybe something else...

03.04.2024
- fixed paths with thanks to [@w-e-w](https://github.com/w-e-w)
- added to Extensions list of Automatic1111

30.03.2024
- added img2img
- added clip_skip option
- comact interface
- added random seed button
- added option for multidecoder
- redesign
- added `torch_dtype=torch.float16` for VAEs
- other optimizations

29.03.2024
- Removed HuggingFace because in a1111 nobody use it
- added VAE support
- added LoRA and lora_scale(weight) support
- random seed on load
- other optimizations

28.03.2024
- Added support for stable diffusion files
- Added support for custom HuggingFace models
- Rebuild UI
- something else?
</details>

## Installation:
#### Easy way
Just install it directly from the Extensions tab

#### or (middle way)

Extensions => Install from URL => `https://github.com/sebaxakerhtc/sd-webui-demofusion.git` => Install
Then switch to installed tab, click Apply and restart UI

#### or (hardcore XD)

`git clone https://github.com/sebaxakerhtc/sd-webui-demofusion.git` from commandline inside Extensions folder

## Usage:
Just use your favorite resolutions with random or custom seed, and when you like the image you see,
change Scale factor, Additional settings (if you need) and click Generate button. Wait for the result. That's it!

To do:
- ~~img2img~~
- ~~LoRA support~~
- ControlNet, maybe...
- ~~Custom resolutions~~
- SD 1.5 adaptation? Soon...
