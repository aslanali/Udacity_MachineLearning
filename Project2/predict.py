# standard modules

import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

#User defined modules
from data_loadr import load_data, process_image
from classifier_functions import load_checkpoint, predict, test_model

parser = argparse.ArgumentParser(description='Using neural network to make prediction')

parser.add_argument('--image_path', action='store',
                    default = 'flowers/test/73/image_00401',
                    help='path to image')

parser.add_argument('--save_dir', action='store',
                    dest='save_dir', default = 'project2.pth',
                    help='location to save checkpoint')

parser.add_argument('--arch', action='store',
                    dest='pretrained_model', default='vgg11',
                    help='pretrained model to use, default is VGG-11')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 5,
                    help='number of top most likely classes to view, default is 5')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='path to image')

parser.add_argument('--gpu', action="store_true", default=False,
                    help='Turn GPU mode on/off, default is off')

results = parser.parse_args()
import os 
print(os.getcwd())
save_dir = results.save_dir
image = results.image_path
top_k = results.topk
gpu_mode = results.gpu
cat_names = results.cat_name_dir
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

# Get pretrained model
pretrained_model= results.pretrained_model
model = getattr(models,pretrained_model)(pretrained=True)

# Load model
loaded_model = load_checkpoint(model, save_dir, gpu_mode)

# Preprocess image - assumes jpeg format
processed_image = process_image(image)

if gpu_mode == True:
    processed_image = processed_image.to('cuda')
else:
    pass

#predict based on model
probs, classes = predict(processed_image, loaded_model, top_k, gpu_mode)

# Print the probabilities and predicted classes
print(probs)
print(classes) 

names = []
for i in classes:
    names += [cat_to_name[i]]
    
# Print the name of predicted flower with highest probability
print(f"Based on prediction the most likely flower is : '{names[0]}' with a probability of {round(probs[0]*100,2)}% ")