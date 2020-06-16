import argparse
import json
import numpy as np

import torch

from PIL import Image

from project_utils import load_data, ArchName, create_device

def get_args():
    parser = argparse.ArgumentParser(description='Image Classifier Project Trainer')
    parser.add_argument('input', action="store", help = 'Enter path of an image to predict')
    parser.add_argument('checkpoint', action="store", help = 'Enter path of checkpoint file')
    
    parser.add_argument('--top_k', action="store", type=int, default = 1, help = 'Enter K to show the top K most likely classes (Default: 1)')
    parser.add_argument('--category_names', action="store", default = 'cat_to_name.json', help = 'Enter the path of mapping file for assigning classes to real names (Default: cat_to_name.json)')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for predicting')

    return parser.parse_args()

def load_model (file_path):
    checkpoint = torch.load(file_path)

    model = ArchName(checkpoint['arch_name']).load_model()        
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs_count = checkpoint['epochs_count']
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_image = Image.open(image)
    
    width, height = pil_image.width, pil_image.height
    if width < height:
        w_percent = (256/float(width))
        width = 256
        height = int(float(height)*w_percent)
    else:
        h_percent = (256/float(height))
        width = int(float(width)*h_percent)
        height = 256
    pil_image = pil_image.resize((width, height), Image.ANTIALIAS)
    
    crop_size = 224
    crop_left = abs((width // 2) - (crop_size // 2))
    crop_top = abs((height // 2) - (crop_size // 2))
    pil_image = pil_image.crop((crop_left, crop_top, crop_left + crop_size, crop_top + crop_size))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = np.array(pil_image)
    np_image = ((np_image / 255) - mean) / std
    
    np_image= np_image.transpose (2,0,1)
    
    return np_image

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    np_image = process_image(image_path)
    t_image = torch.from_numpy(np_image)
    t_image = t_image.unsqueeze_(0)
    t_image = t_image.type(torch.FloatTensor)
    t_image = t_image.to(device)
    
    model.eval()
    with torch.no_grad():
        logps = model(t_image)
    model.train()
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    predictions = top_p[0].cpu().numpy()
    
    idx_to_class = {v:k for k,v in model.class_to_idx.items()}
    
    predicted_classes = np.array([idx_to_class[idx] for idx in top_class[0].cpu().numpy()])
    
    return predictions, predicted_classes

def main():
    args = get_args()
    top_k = args.top_k

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    device = create_device(args.gpu)
    model = load_model(args.checkpoint)
    model.to(device)

    top_p, top_class = predict(args.input, model, device, top_k)
    top_class_name = [cat_to_name[i] for i in top_class]

    for index in range(0, top_k):
        print(f'TOP {index + 1}/{top_k}.. Name: {top_class_name[index]}; Probability: {top_p[index]}..')

if __name__ == "__main__":
    main()