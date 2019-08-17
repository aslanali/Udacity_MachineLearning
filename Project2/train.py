#standard modules

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
#User defined modules

from data_loadr import load_data
from classifier_functions import build_classifier, validation, train_model, test_model, save_model, load_checkpoint

parser = argparse.ArgumentParser(description='Training the neural network')

parser.add_argument('--data_dir', action = 'store',help = 'path to training data.',dest ='data_dir')

parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg11',
                    help= 'pretrained model to use; The default is VGG-11')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_dir', default = 'project2.pth',
                    help = 'location to save checkpoint in')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.001,
                    help = 'learning rate for training the model, default is 0.001')

parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=int, default = 0.1,
                    help = 'dropout for training the model, default is 0.1')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=int, default = 300,
                    help = 'number of hidden units in classifier, default is 300')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 7,
                    help = 'number of epochs to use during training, default is 7')

parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off, default is off.')

results = parser.parse_args()
#parsing all arguments with respect to training 

data_dir = results.data_dir
save_dir = results.save_dir
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.units
epochs = results.num_epochs
gpu_mode = results.gpu

# Load and transform the image data according to model  
trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)

# Load the pretrained model
pretrained_model = results.pretrained_model
model = getattr(models,pretrained_model)(pretrained=True)

# Build and attach new classifier
input_units = model.classifier[0].in_features
build_classifier(model, input_units, hidden_units, dropout)

#  NLLLoss when using Softmax is a good option
criterion = nn.NLLLoss()
# Adam optimizer uses momentum which can help overcome local minima in comparison to SGD
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# Train model
model, optimizer = train_model(model, epochs,trainloader, validloader, criterion, optimizer, gpu_mode)

# Test model
test_model(model, testloader, gpu_mode)
# Save model
save_model(model, train_data, optimizer, save_dir, epochs)