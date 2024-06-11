from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers.utils import load_image
import json
import requests
import base64
import os
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

token = os.environ.get("HF_TOKEN")

class Item(BaseModel):
    prompt: str
    steps: int
    guidance: float
    modelID: str
    image: str
    scale: str   # Change this to Accomodate a Dict type with a Dict[List] When IP Adapter is Availab

class promptType(BaseModel):
    prompt: str
    modelID: str

@app.post("/inferencePrompt")
async def inferencePrompt(item: promptType):
    input = item.prompt
    modelID = item.modelID
    API_URL = f'https://api-inference.huggingface.co/models/{modelID}'
    print(API_URL)
    headers = {"Authorization": f"Bearer {token}"}    
    print(headers)
    parameters = {"return_full_text":False,"max_new_tokens":500}
    options = {"use_cache": False, "wait_for_model": True}
    response = requests.post(API_URL, headers=headers, \
        json={"inputs":input, "parameters": parameters,"options": options})
    print(response.json())
    if response.status_code != 200:
        print(response.json().get("error_type"), response.status_code)
        return {"error": response.json().get("error")}
    
    return response.json()

@app.post("/api")
async def inference(item: Item):
    print(item.prompt)
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
        prompt = "voxel style, " + item.prompt
    if "BalloonArt" in item.modelID:
        prompt = "BalloonArt, " + item.prompt
    if "PaperCut" in item.modelID:
        prompt = "PaperCut, " + item.prompt

    negative_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
    data = {"inputs":prompt, "negative_prompt": negative_prompt, "options":{"wait_for_model": True, "use_cache": False}}
    API_URL = "https://api-inference.huggingface.co/models/" + item.modelID
    headers = {"Authorization": f"Bearer {token}"}
    api_data = json.dumps(data)
    response = requests.request("POST", API_URL, headers=headers, data=api_data)
    
    image_stream = BytesIO(response.content)
    image = Image.open(image_stream)
    image.save("response.png")
    with open('response.png', 'rb') as f:
        base64image = base64.b64encode(f.read())
    
    return {"output": base64image}

app.mount("/", StaticFiles(directory="web-build", html=True), name="build")

@app.get('/')
def homepage() -> FileResponse:
    return FileResponse(path="/app/build/index.html", media_type="text/html")

    #  IP Adapter API Integration
    '''
    negative_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"

    image_stream = None
    if "refiner" in item.modelID:
        image = item.image.replace('data:image/png;base64,','')
        image_data = base64.b64decode(image)
        image_stream = BytesIO(image_data)
        image = Image.open(image_stream)
        image.save("ip.png")

        # New code to convert the image to bytes
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        image_stream = image_bytes

    if 'up' in item.scale.keys() :
        print('scaled')
        #data["set_ip_adapter_scale"] = item.scale
    
    data = {
        "inputs": prompt, 
        "negative_prompt": negative_prompt, 
        "ip_adapter_image": load_image('https://digital-photography-school.com/is-unsplash-really-an-issue-for-photographers/'),
        "options": {"wait_for_model": True, "use_cache": False}
    }
    API_URL = "https://api-inference.huggingface.co/models/" + item.modelID
    headers = {"Authorization": f"Bearer {token}"}

    
    # Modified request to use image_bytes
    response = requests.post(API_URL, headers=headers, data=data )


    print(f'RESPONSE:  {response.json()}')
    if response.status_code == 200 and 'image' in response.headers['Content-Type']:
        
        image_stream = BytesIO(response.content)
        image = Image.open(image_stream)
        image.save("response.png")
        
        with open('response.png', 'rb') as f:
            base64image = base64.b64encode(f.read())
        
        return {"output": base64image}
    else:
        return {"error": "Invalid response from API"}
    '''
