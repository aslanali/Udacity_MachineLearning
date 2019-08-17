import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms, models

#load and preprocess the data
def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    # The validation set will use the same transform as the test set
    test_transforms = transforms.Compose([transforms.Resize(250),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 

    
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=60, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=60)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=60)
    
    return trainloader, testloader, validloader, train_data, test_data, valid_data


#loading and preprocess test image
def process_image(image):
    ''' 
    returns an Numpy array
    '''
    # Converting image to PIL image using image file path
    im = Image.open(f'{image}' + '.jpg')

    # Building image transform
    transform = transforms.Compose([transforms.Resize(250),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    ## Transform image 
    im_transformed = transform(im)
    
    # Converting to Numpy array 
    im_transformed_np = np.array(im_transformed)
    
    # Converting to torch tensor from Numpy array
    imgtensor = torch.from_numpy(im_transformed_np).type(torch.FloatTensor)

    img2 = imgtensor.unsqueeze_(0)
    
    return img2