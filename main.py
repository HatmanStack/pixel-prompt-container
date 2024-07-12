from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoPipelineForText2Image, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import InferenceClient, login
from transformers import AutoTokenizer, T5Tokenizer
from pydantic import BaseModel
from gradio_client import Client, file
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
from dotenv import load_dotenv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

load_dotenv()
token = os.environ.get("HF_TOKEN")
login(token)
prompt_model = "meta-llama/Meta-Llama-3-70B-Instruct"
prompt_model_backup = "mistralai/Mistral-7B-Instruct-v0.3"
magic_prompt_model = "Gustavosta/MagicPrompt-Stable-Diffusion"
options = {"use_cache": False, "wait_for_model": True}
parameters = {"return_full_text":False}
headers = {"Authorization": f"Bearer {token}", "x-use-cache":"0", 'Content-Type' :'application/json'}
API_URL = f'https://api-inference.huggingface.co/models/'
perm_negative_prompt = "watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"

class Item(BaseModel):
    prompt: str
    steps: int
    guidance: float
    modelID: str
    image: str
    scale: Dict[str, Dict[str, List[float]]]    

class Core(BaseModel):
    itemString: str

@app.post("/core")
async def core(item: Core):
    print(item.itemString)
    #wake_model(item.model)

def getPrompt(prompt, modelID, attempts=1):
    input = prompt
    if modelID != magic_prompt_model:
        tokenizer = AutoTokenizer.from_pretrained(modelID)
        chat = [
            {"role": "user", "content": prompt_base},
            {"role": "assistant", "content": "What is your seed string?"},
            {"role": "user", "content": prompt},
            ]
        input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    try:
        apiData={"inputs":input, "parameters": parameters, "options": options}
        response = requests.post(API_URL + modelID, headers=headers, data=json.dumps(apiData))
        if response.status_code == 200:
            try:
                responseData = response.json()
                return responseData
            except ValueError as e:
                print(f"Error parsing JSON: {e}")
        else:
            print(f"Error from API: {response.status_code} - {response.text}")
            if attempts < 3:
                modelID = prompt_model if modelID == prompt_model_backup else prompt_model_backup
                getPrompt(prompt, modelID, attempts + 1)
    except Exception as e:
        print(f"An error occurred: {e}")
        if attempts < 3:
            modelID = prompt_model if modelID == prompt_model_backup else prompt_model_backup 
            getPrompt(prompt, modelID, attempts + 1)
    return response.json()
    
@app.post("/inferencePrompt")
async def inferencePrompt(item: Core):
    plain_response_data = getPrompt(item.itemString, prompt_model)
    magic_response_data = getPrompt(item.itemString, magic_prompt_model)
    returnJson = {"plain": plain_response_data[0]["generated_text"], "magic": item.itemString + magic_response_data[0]["generated_text"]}
    return returnJson

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

def gradioSD3(item):
    client = Client(item.modelID, hf_token=token)
    result = client.predict(
            prompt=item.prompt,
            negative_prompt=perm_negative_prompt,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=item.guidance,
            num_inference_steps=item.steps,
            api_name="/infer"
    )
    
    img = Image.open(result[0])
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_img = base64.b64encode(img_byte_arr)
    
    return base64_img

def gradioRefiner(item):
    image_data = base64.b64decode(item.image)
    # Save the image to a file
    with open('image.png', 'wb') as f:
        f.write(image_data)
    client = Client("tonyassi/IP-Adapter-Playground", hf_token=token)
    result = client.predict(
            ip=file('image.png'),
            prompt=item.prompt,
            neg_prompt=perm_negative_prompt,
            width=1024,
            height=1024,
            ip_scale=item.scale,
            strength=0.7,
            guidance=item.guidance,
            steps=item.steps,
            api_name="/text_to_image"
    )
    
    img = Image.open(result[0])
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_img = base64.b64encode(img_byte_arr)
    
    return base64_img
    

@app.post("/api")
async def inference(item: Item):
    if "stable-diffusion-3" in item.modelID:
        useGradio = gradioSD3(item)
        return {"output": useGradio}
    if "gradiotxt2img" in item.modelID:
        useGradio = gradioRefiner(item)
        return {"output": useGradio}
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
    
    data = {"inputs":tokenizer.decode(tokenized_prompt_list), "negative_prompt": perm_negative_prompt, "options":options}
    
    api_data = json.dumps(data)
    response = requests.post( API_URL + item.modelID, headers=headers, data=api_data)

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
            negative_prompt=perm_negative_prompt,
            guidance_scale=item.guidance,
            num_inference_steps=item.steps,
            generator=generator,
        ).images[0]
        image.save("response.png")

        with open('response.png', 'rb') as f:
            base64image = base64.b64encode(f.read())
        
        return {"output": base64image}
    


prompt_base = 'Instructions:\
\
1. Take the provided seed string as inspiration.\
2. Formulate a single-sentence prompt that is clear, vivid, and imaginative.\
3. Ensure the prompt is between 50 and 77 tokens.\
4. Return only the prompt.\
Format your response as follows:\
Stable Diffusion Prompt: [Your prompt here]\
\
Remember:\
\
- The prompt should be concise yet descriptive.\
- Avoid overly complex or abstract phrases.\
- Make sure the prompt evokes strong imagery and can guide the creation of visual content.'


app.mount("/", StaticFiles(directory="web-build", html=True), name="build")

@app.get('/')
def homepage() -> FileResponse:
    return FileResponse(path="/app/build/index.html", media_type="text/html")