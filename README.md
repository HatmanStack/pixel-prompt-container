# Pixel Prompt Container :zap:

This repository contains a static React Native application built using Expo with FastApi and Docker for deployment. It's a single container that's running [here](https://huggingface.co/spaces/Hatman/pixel-prompt).  It's serving several diffusion models that use the huggingface [inference-api](https://huggingface.co/docs/api-inference/index). The root repository and alternative deployments are [here](https://github.com/HatmanStack/pixel-prompt). A blog post explaining this deployment and the HuggingFace Inference API can be found [here](https://medium.com/@HatmanStack/cloud-bound-hugging-face-spaces-1101c569690d).

## Installation :octocat:

To generate the static content for this container have a working Node/npm installation and clone the [pixel-prompt-frontend](https://github.com/HatmanStack/pixel-prompt-frontend) repo.  Run these commands in the pixel-prompt-frontend root directory to generate static content.

```shell
npm install -g yarn
yarn
npx expo export:web
```

Static files are output to the web-build folder in the root directory. Move the web-build from your pixel-prompt-frontend root directory to your pixel-prompt-container root directory. **To reach the endpoint from the frontend use `/api` in the pixel-prompt-frontend NOT `http://localhost:7860/api`** 

Add your HF_TOKEN variable as an environmental variable in your container settings.

## License

This project is licensed under the [MIT License](LICENSE)

## Acknowledgments :trophy:

- This application is built with Expo, a powerful framework for building cross-platform mobile applications. Learn more about Expo: [https://expo.io](https://expo.io)

