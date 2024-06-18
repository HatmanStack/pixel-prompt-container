---
title: Pixel Prompt
emoji: 🔥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
---
# Pixel Prompt Container 

This repository contains a static React Native application built using Expo with FastApi and Docker for deployment.   It's serving several diffusion models that use the huggingface [inference-api](https://huggingface.co/docs/api-inference/index). A blog post explaining this deployment and the HuggingFace Inference API can be found [here](https://medium.com/@HatmanStack/cloud-bound-hugging-face-spaces-1101c569690d).

## Implementation 

You can see other implementations of this application [here](https://github.com/hatmanstack/pixel-prompt).

## Installation   💻

To generate the static content for this container have a working Node/npm installation and clone the [pixel-prompt-frontend](https://github.com/HatmanStack/pixel-prompt-frontend) repo.  Run these commands in the pixel-prompt-frontend root directory to generate static content.

```shell
npm install -g yarn
yarn
npx expo export:web
```

Static files are output to the web-build folder in the root directory. Move the web-build from your pixel-prompt-frontend root directory to your pixel-prompt-container root directory. **To reach the endpoint from the frontend use `/api` in the pixel-prompt-frontend NOT `http://localhost:7860/api`** 

Add your HF_TOKEN variable as an environmental variable in your container settings.

## Models   ⚡

All models are opensource and available on HuggingFace.

### Diffusion

#### Image to Image

- **timbrooks/instruct-pix2pix**
- **stabilityai/stable-diffusion-xl-refiner-1.0**
       
#### Text to Image

- **stabilityai/stable-diffusion-3-medium**
- **stabilityai/stable-diffusion-xl-base-1.0**
- **SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep**
- **Fictiverse/Fictiverse/Stable_Diffusion_VoxelArt_Model**
- **Fictiverse/Stable_Diffusion_PaperCut_Model**
- **dallinmackay/Van-Gogh-diffusion**
- **nousr/robo-diffusion**
- **Eugeoter/artiwaifu-diffusion-1.0**
- **nitrosocke/Arcane-Diffusion**
- **Fictiverse/Stable_Diffusion_BalloonArt_Model**
- **prompthero/openjourney**
- **juliajoanna/sdxl-flintstones_finetuning_1**
- **segmind/Segmind-Vega**
- **digiplay/AbsoluteReality_v1.8.1**
- **dreamlike-art/dreamlike-photoreal-2.0**
- **digiplay/Acorn_Photo_v1**

### Prompts

- **mistralai/Mistral-7B-Instruct-v0.3**
- **roborovski/superprompt-v1**
- **google/gemma-1.1-7b-it**

## Functionality

This App was creating using the HuggingFace Inference API.  Although Free to use, some functionality isn't available yet.  The Style and Layout switches are based on the IP adapter which isn't supported by the Inference API. If you decide to use custom endpoints this is available now.

## License

This project is licensed under the [MIT License](LICENSE)

## Acknowledgments   🏆

This application is built with Expo, a powerful framework for building cross-platform mobile applications. Learn more about Expo: [https://expo.io](https://expo.io)

This application is using the HuggingFace Inference API, provided by <a href="https://huggingface.co">HuggingFace</a> 
