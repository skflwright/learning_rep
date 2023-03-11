# import fastapi
from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse
from diffusers import (
    StableDiffusionPipeline, # huggingface pipeline
    StableDiffusionImg2ImgPipeline
)
from PIL import Image
import numpy as np
from torch import autocast
import uvicorn
import cv2
import io
import torch

# instantiate the app
app = FastAPI() #first thing you need to do is instantiate your FastApp - this object is going to
# be a FastAPI object

# cuda or cpu config  - this is a helper function. This tells our program to either use cuda
# if cuda is available (cuda basically means GPU) so it will iterate through the images much faster
# but if no GPU is available, we can just use the cpu. The helper function enables us to not have
# to choose or hard code that choice.
def get_device():
    if torch.cuda.is_available():
        print('cuda is available')
        return torch.device('cuda')
    else:
        print('cuda is not available')
        return torch.device('cpu')

# create a route - this is the meat of the application.  We are going to create a fast end-point
# by using this handy notation (@app.post("/text2img"))
@app.get("/")
def index():
    return {"text" : "We're running!"}

# create a text2img route
@app.post("/text2img") # this is called a decorator, signified by the @ symbol, which wraps the 
# function 'text:str' in a bunch of additional code  and the we're using the app.post (which is our API app) and we're 
# creating a post end-pointand naming it text2img.  The decorator notation in python is helpful and pydantic because 
# it lets us have all that additional code without having to write it or surround our function.

# The difference between post and get is that a post endpoint is going to expect us to send it something
# and then it will return a response, versus a get end point which would just return a response if we pinged it. 
def text2img(text: str): # this is the work function. It accepts a text, which it expects to be in a string  and then we have our device
    #so we just use our helper function to that. 
    device = get_device()
# then we have our model. We've already downloaded the weights from huggingface and we have them locally in the model folder so that
# we don't have to download the weights every time we run this function. Addtionally, we're loading the pipeline from huggingface. 
# Then we are going to cast our pipeline to to our device (text2img_pipe.to(device)...whichever it finds. After that, all we need to do is
# run our pipeline on the text prompt (img=text2img_pipe(text).images[0]). We're going to get the 0th or first image and save that into 
# our image variable. 
    text2img_pipe = StableDiffusionPipeline.from_pretrained("../model")
    text2img_pipe.to(device)
#afterwards we need to do some post processing to put it into a format that works with swagger (fastAPI's frontend).
    img = text2img_pipe(text).images[0]

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res, img = cv2.imencode(".png", img)
#  then we're going to do some life cycle management in the end point (del text2img_pipe). The reason we are doing this is because you'll notice 
# that we are actually getting the pipeline in the end point to make the demo more accessible to people so you don't have to have huge resources to 
# be able to run it.  Because of that, we want to make sure we are doing some life cycle management to make sure we are not using too much resources.  
    del text2img_pipe # this empties out the cache, which is good practice. 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# Lastly we are going to return our image to swagger (io.BytesIO(img.tobytes) in the format it expects and we tell it what it should look like (media_type='image/png') and 
# FastAPI does the rest. Swagger UI as an OpenAPI.
    return StreamingResponse(io.BytesIO(img.tobytes()), media_type="image/png")

# run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # this last piece of code is just boiler plate (if name = main), which means when we run the file itself, we're going to use uvicorn, an ASGI server 
    # to run our app , host it on our local host and expose it on port 8000. So all of this comes together to let us use  the application. c