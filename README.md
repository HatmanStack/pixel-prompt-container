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

## Preview :zap:

To preview the application visit the hosted version on the Hugging Face Spaces platform [here](https://huggingface.co/spaces/Hatman/pixel-prompt).

## Installation :hammer:

To generate the static content for this container have a working Node/npm installation and clone the [pixel-prompt-frontend](https://github.com/HatmanStack/pixel-prompt-frontend) repo.  Run these commands in the pixel-prompt-frontend root directory to generate static content.

```shell
npm install -g yarn
yarn
npx expo export:web
```

Static files are output to the web-build folder in the root directory. Move the web-build from your pixel-prompt-frontend root directory to your pixel-prompt-container root directory. **To reach the endpoint from the frontend use `/api` in the pixel-prompt-frontend NOT `http://localhost:7860/api`** 

Add your HF_TOKEN variable as an environmental variable in your container settings.

## Models :sparkles:

All models are opensource and available on HuggingFace.

### Diffusion

- **stabilityai/stable-diffusion-xl-base-1.0**
- **stabilityai/stable-diffusion-xl-refiner-1.0**
- **prompthero/openjourney**
- **dreamlike-art/dreamlike-photoreal-2.0**
- **nitrosocke/Arcane-Diffusion**
- **dallinmackay/Van-Gogh-diffusion**
- **nousr/robo-diffusion**

### Prompts

- **mistralai/Mistral-7B-Instruct-v0.3**

## Functionality

The Style and Layout components use Huggingface's IP adapter which isn't available in the HuggingFace Inference API yet.  If you decide to use custom endpoints this is available now.  A placeholder image is included.

## License

This project is licensed under the [MIT License](LICENSE)

## Acknowledgments :trophy:

This application is built with Expo, a powerful framework for building cross-platform mobile applications. Learn more about Expo: [https://expo.io](https://expo.io)

<p align="center">This application is using the HuggingFace Inference API, provided by <a href="https://huggingface.co">HuggingFace</a> </br><img src="https://github.com/HatmanStack/pixel-prompt-backend/blob/main/logo.png" alt="Image 4"></p>

