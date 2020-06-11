import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image

import json


def parse():
    
    parser = argparse.ArgumentParser(description = 'Predict output of the neural network')
    parser.add_argument('--image', default = './flowers/test/12/image_04012.jpg', help = 'State the path of the image to classify.')
    parser.add_argument('--checkpoint', default= '.', help = 'Specify the path to the checkpoint file.')
    parser.add_argument('--category_names', default = './cat_to_name.json', help = 'Specify the mapping of categories to labels (real names)')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Set the number of most likely classes to predict')                 parser.add_argument('--gpu', default = 'cpu', help = 'Set GPU for inference as "gpu"(recommended), default = "cpu"')
    
    args = parser.parse_args()
    
    
    return args



def load_checkpoint(checkpoint):
    """ Load the checkpoint to get our previously trained model """
    
    checkpoint = torch.load(args.checkpoint)
    
    model = models.__dict__[checkpoint['architecture']](pretrained=True)
    epochs = checkpoint['epochs']
    model.features = checkpoint['features']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array """
    
    # loading the image using PIL module
    loaded_image = Image.open(args.image)
    
    # Transforms needed for Image Processing 
    processing_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Processing the loaded image
    image_as_tensor = processing_transforms(loaded_image)
    
    return image_as_tensor



def imshow(image, ax = None, title = None):
    """ Imshow for Tensor. """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = args.image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model. """
    
    # TODO: Implement the code to predict the class from an image file
    processed_image = process_image(image_path)
    
    model.to(device)
    
    
    with torch.no_grad():
        
        processed_image = processed_image.unsqueeze_(0).to(device)
        
        log_probabilities = model(processed_image)
        probabilities = torch.exp(log_probabilities)
        top_probabilities, top_class = probabilities.topk(args.topk, dim = 1)
        
        
        top_probas, top_classes = top_probabilities.data.cpu().numpy()[0], top_class.data.cpu().numpy()[0]
        
        idx_to_class = dict(map(reversed, model.class_to_idx.items()))
        top_classes = [idx_to_class[top_classes[i]] for i in range(top_classes.size)]

        return top_probas, top_classes
    

    
def plot_topk(image_path, model):
    """ Display the image along with its topk classes """
    
    # get top 5 classes for the image
    top_probabilities, top_classes = predict(image_path, model)
    
    # load image using PIL module
    loaded_image = Image.open(image_path)
    
    # get flowers' classes (their name, not indices)
    flowers_classes = [cat_to_name[i] for i in top_classes]
    
    # plotting 
    fig, ax = plt.subplots(2,1)  
    
    ax[0].imshow(loaded_image);
    
    y_ticks = np.arange(len(top_classes))
    plt.barh(y_ticks, top_probabilities)
    plt.yticks(y_ticks, flowers_classes)
    
    plt.show()
    



def main(image_path, checkpoint, category_names):
    
    model = load_checkpoint(args.checkpoint)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    image_path = args.image
    
    
    plot_topk(image_path, model)

    
    
if __name__ == '__main__':
    main()
    


