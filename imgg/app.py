import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import os
from diffusers import StableDiffusionPipeline
import torch
import logging

LOG_PATH = './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_image.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, 
                   format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')

current_model = None

def load_model(model_id, device, torch_dtype):
    try:
        global current_model
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_model] trying to load model: {model_id}')
        
        if current_model is None:
            current_model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            ).to(device)
            
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_model] [success] Model loaded!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_model] [error] Failed to load model: {e}')
        raise

def generate_image(model_id, prompt, device, torch_dtype, output_path="generated_image.png"):
    try:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to load model: {model_id}')
        load_model(model_id, device, torch_dtype)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] Model loaded!')
        
        start_time = time.time()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] generating image for prompt: {prompt}')
        
        image = current_model(prompt).images[0]
        image.save(output_path)
        
        processing_time = time.time() - start_time
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] finished generating image! Saved to {output_path} in {processing_time:.2f}s')
        
        return {
            "output_path": output_path,
            "processing_time": f"{processing_time:.2f}s",
            "status": "success"
        }
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] [error]: {e}')
        return {
            "error": str(e),
            "status": "failed"
        }

redis_connection = None

def start_redis(req_redis_port):
    try:
        r = redis.Redis(host="redis", port=req_redis_port, db=0)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Redis started successfully.')
        return r
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Failed to start Redis on port {req_redis_port}: {e}')
        raise

app = FastAPI()

@app.get("/")
async def root():
    return 'Hello from image generation server!'

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """
    Serve generated PNG images.
    Example: /images/generated_image.png
    """
    image_path = f"./{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    if not image_name.lower().endswith('.png'):
        raise HTTPException(status_code=400, detail="Only PNG files are supported")
    return FileResponse(image_path, media_type="image/png")

@app.post("/generate")
async def generate(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        
        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": "ok"})
            
        if req_data["method"] == "generate_image":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to generate image...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to generate image...')
            
            result = generate_image(
                req_data["model_id"],
                req_data["prompt"],
                req_data["device"],
                eval(req_data["torch_dtype"]),  # Converts string to torch.dtype
                req_data.get("output_path", "generated_image.png")
            )
            
            return JSONResponse({"result_status": 200, "result_data": result})
            
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host=f'{os.getenv("IMGG_IP")}', port=int(os.getenv("IMGG_PORT")))