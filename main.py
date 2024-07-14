import random
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient, login
from transformers import AutoTokenizer
from pydantic import BaseModel
from gradio_client import Client
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
from datetime import datetime
import torch
import json
import requests
import base64
import os
from typing import Dict, List  
from PIL import Image
from io import BytesIO
import aiohttp
import asyncio
from dotenv import load_dotenv
import boto3


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
cwd = os.getcwd()
pictures_directory = os.path.join(cwd, 'pictures')

class Item(BaseModel):
    prompt: str
    steps: int
    guidance: float
    modelID: str
    modelLabel: str
    image: str
    scale: Dict[str, Dict[str, List[float]]]    

class Core(BaseModel):
    itemString: str

@app.get("/core")
async def core():
    if not os.path.exists(pictures_directory):
        os.makedirs(pictures_directory)
    base64_data = []  # List to hold parsed data from each JSON file
    prompt_data = []
    # Iterate through each file in the directory
    for filename in os.listdir(pictures_directory):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            file_path = os.path.join(pictures_directory, filename)
            
            # Open and parse the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                base64_data.append(data["base64image"])
                prompt_data.append(data["returnedPrompt"])
    return {"base64": base64_data, "prompt": prompt_data}
    

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
    base64_img = base64.b64encode(img_byte_arr).decode('utf-8')
    
    save_image(base64_img, item)
    return base64_img

def gradioIPAdapter(item):
    client = Client('Hatman/12038975', hf_token=token)
    result = client.predict(
            scale=item.scale,
            image=item.image,
            prompt=item.prompt,
            negative_prompt=perm_negative_prompt,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=item.guidance,
            num_inference_steps=item.steps,
            api_name="/pixel_prompt_ip_adapter"
    )
    
    img = Image.open(result[0])
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_img = base64.b64encode(img_byte_arr).decode('utf-8')
    
    save_image(base64_img, item)
    return base64_img

@app.post("/api")
async def inference(item: Item):
    prompt = item.prompt
    activeModels = InferenceClient().list_deployed_models()
    #if "none" not in item.scale.keys():
        #base64_img = gradioIPAdapter(item)
    if "stable-diffusion-3" in item.modelID:
        useGradio = gradioSD3(item)
        return {"output": useGradio}
    elif "Voxel" in item.modelID or "pixel" in item.modelID:
        if "Voxel" in item.modelID:
            prompt = "voxel style, " + item.prompt
        base64_img = lambda_image(prompt, item.modelID)
    elif item.modelID not in activeModels['text-to-image']:
        asyncio.create_task(wake_model(item.modelID))
        return {"output": "Model Waking"}  
    else:
        print("Model active")
        if "dallinmackay" in item.modelID:
            prompt = "lvngvncnt, " + item.prompt
        data = {"inputs":prompt, "negative_prompt": perm_negative_prompt, "options":options}
        api_data = json.dumps(data)
        response = requests.request("POST", API_URL + item.modelID, headers=headers, data=api_data)
        image_stream = BytesIO(response.content)
        image = Image.open(image_stream)
        image.save("response.png")
        with open('response.png', 'rb') as f:
            base64_img = base64.b64encode(f.read()).decode('utf-8')

    save_image(base64_img, item)

    return {"output": base64_img}

def save_image(base64image, item):
    data = {"base64image": "data:image/png;base64," + base64image, "returnedPrompt": item.modelLabel + " " + item.prompt, "prompt": item.prompt, "steps": item.steps, "guidance": item.guidance, "scale": item.scale}
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(pictures_directory, f'{timestamp}.json')
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def lambda_image(prompt, modelID):
    data = {
    "prompt": prompt,
    "modelID": modelID
    }
    serialized_data = json.dumps(data)
    aws_id = os.environ.get("AWS_ID")
    aws_secret = os.environ.get("AWS_SECRET")
    aws_region = os.environ.get("AWS_REGION")
    session = boto3.Session(aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=aws_region)
    lambda_client = session.client('lambda')
    response = lambda_client.invoke(
        FunctionName='pixel_prompt_lambda',
        InvocationType='RequestResponse',  
        Payload=serialized_data  
    )
    response_payload = response['Payload'].read()
    response_data = json.loads(response_payload)

    return response_data['body']

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