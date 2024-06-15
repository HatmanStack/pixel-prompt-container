from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoPipelineForText2Image, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, T5Tokenizer
from pydantic import BaseModel
import torch
import json
import requests
import base64
import os
from typing import Dict, List  
from PIL import Image, ImageOps
from io import BytesIO
import aiohttp
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

token = os.environ.get("HF_TOKEN")

options = {"use_cache": False, "wait_for_model": True}
headers = {"Authorization": f"Bearer {token}", "x-use-cache":"0"}
API_URL = f'https://api-inference.huggingface.co/models/'

class Item(BaseModel):
    prompt: str
    steps: int
    guidance: float
    modelID: str
    image: str
    scale: Dict[str, Dict[str, List[float]]]    

class PromptType(BaseModel):
    prompt: str
    modelID: str

class Core(BaseModel):
    model: str

@app.post("/core")
async def core(item: Core):
    wake_model(item.model)

def getMistrailPrompt(prompt, modelID, max_tokens=1000, attempts=1):
    modelID = "mistralai/Mistral-7B-Instruct-v0.3" if modelID == 'google/gemma-1.1-7b-it' else "google/gemma-1.1-7b-it"   
    tokenizer = AutoTokenizer.from_pretrained(modelID)
    chat = []
    if modelID == "mistralai/Mistral-7B-Instruct-v0.3":
        chat = [
            {"role": "user", "content": "You create prompts for the Stable Diffusion series of machine learning models."},
            {"role": "assistant", "content": "What would you like me to do?"},
            {"role": "user", "content": f'Your prompt should be confied to {max_tokens} tokens maximum.  Here is your seed string: {prompt}'},
            ]
    if modelID == "google/gemma-1.1-7b-it":
        chat = [
                {"role": "user", "content": f'You create prompts for the Stable Diffusion series of machine learning models.  \
                Your prompt should be confied to {max_tokens} tokens maximum.  Here is your seed string: {prompt}'}]
    
    input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    parameters = {"return_full_text":False,"max_new_tokens":max_tokens}
    response = requests.post(API_URL + modelID, headers=headers, \
        json={"inputs":input, "parameters": parameters,"options": options})
    if response.status_code != 200:
        print(response.json().get("error_type"), response.status_code)
        return {"error": response.json().get("error")}
    if 'I am unable to provide' in response.json()[0]['generated_text']:
        attempts -= 1
    if attempts < 2:
        getMistrailPrompt(prompt, modelID, max_tokens-700, attempts + 1)
    return response.json()

@app.post("/inferencePrompt")
async def inferencePrompt(item: PromptType):
    response_data = getMistrailPrompt(item.prompt, item.modelID)
    response_data[0]["generated_text"] = response_data[0]["generated_text"].replace("*", "")
    if "error" in response_data:
        return response_data
    else:
        input_text = "Expand the following prompt to add more detail: " + item.prompt.split("seed string. :")[1]
        parameters = {"max_new_tokens":77}
        response1 = requests.post('https://api-inference.huggingface.co/models/roborovski/superprompt-v1', headers=headers, \
            json={"inputs":input_text, "parameters": parameters,"options": options})
        
        if response1.status_code != 200:
            print(response1.json().get("error_type"), response1.status_code)
            return {"error": response1.json().get("error")}
        response_data[0]["flan"] = response1.json()[0]["generated_text"]
        print(response_data)
    return response_data


async def wake_model(modelID):
    data = {"inputs":"wake up call", "options":options}
    headers = {"Authorization": f"Bearer {token}"}
    api_data = json.dumps(data)
    try:
        timeout = aiohttp.ClientTimeout(total=60)  # Set timeout to 60 seconds
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(API_URL + modelID, headers=headers, data=api_data) as response:
                pass
        print('Model Waking')
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
def chunk_prompt(prompt, tokenizer, chunk_size=77):
    tokens = tokenizer.encode(prompt)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks

@app.post("/api")
async def inference(item: Item):
    activeModels = InferenceClient().list_deployed_models()
    if item.modelID not in activeModels['text-to-image']:
        asyncio.create_task(wake_model(item.modelID))
        return {"output": "Model Waking"}
    
    print("Model active")
    prompt = item.prompt
    if "stable-diffusion" in item.modelID:
        prompt = item.prompt
    if "dallinmackay" in item.modelID:
        prompt = "lvngvncnt, " + item.prompt
    if "nousr" in item.modelID:
        prompt = "nousr robot, " + item.prompt
    if "nitrosocke" in item.modelID:
        prompt = "arcane, " + item.prompt
    if "dreamlike" in item.modelID:
        prompt = "photo, " + item.prompt
    if "prompthero" in item.modelID:
        prompt = "mdjrny-v4 style, " + item.prompt 
    if "Voxel" in item.modelID:
        prompt = "VoxelArt, " + item.prompt
    if "BalloonArt" in item.modelID:
        prompt = "BalloonArt, " + item.prompt
    if "PaperCut" in item.modelID:
        prompt = "PaperCut, " + item.prompt

    tokenizer = AutoTokenizer.from_pretrained(item.modelID, subfolder="tokenizer") 
    chunks = chunk_prompt(item.prompt, tokenizer)
    
    tokenized_chunks = [tokenizer.encode(tokenizer.decode(chunk), return_tensors="pt") for chunk in chunks]
    tokenized_prompt = torch.cat(tokenized_chunks, dim=1)
    tokenized_prompt_list = tokenized_prompt.tolist()[0]
    
    negative_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
    data = {"inputs":tokenizer.decode(tokenized_prompt_list), "negative_prompt": negative_prompt, "options":options}
    
    api_data = json.dumps(data)
    response = requests.request("POST", API_URL + item.modelID, headers=headers, data=api_data)

    image_stream = BytesIO(response.content)
    image = Image.open(image_stream)
    image.save("response.png")
    with open('response.png', 'rb') as f:
        base64image = base64.b64encode(f.read())
    
    return {"output": base64image}
    

@app.post("/img2img")
async def img2img(item: Item):
    print()
    if "pix2pix" in item.modelID:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(item.modelID, torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        image_data = base64.b64decode(item.image.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        
        images = pipe(item.prompt, image=image, num_inference_steps=item.steps, image_guidance_scale=item.guidance).images
        
        image_stream = BytesIO(images[0])
        image = Image.open(image_stream)
        image.save("response.png")
        with open('response.png', 'rb') as f:
            base64image = base64.b64encode(f.read())
        
        return {"output": base64image}
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        if "up" in item.scale or "down" in item.scale:
            pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
            pipeline.set_ip_adapter_scale(item.scale)

        image_data = base64.b64decode(item.image)
        image = Image.open(BytesIO(image_data))

        generator = torch.Generator(device="cpu").manual_seed(99)
        image = pipeline(
            prompt=item.prompt,
            ip_adapter_image=image,
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            guidance_scale=item.guidance,
            num_inference_steps=item.steps,
            generator=generator,
        ).images[0]
        image.save("response.png")

        with open('response.png', 'rb') as f:
            base64image = base64.b64encode(f.read())
        
        return {"output": base64image}
    
    
app.mount("/", StaticFiles(directory="web-build", html=True), name="build")

@app.get('/')
def homepage() -> FileResponse:
    return FileResponse(path="/app/build/index.html", media_type="text/html")