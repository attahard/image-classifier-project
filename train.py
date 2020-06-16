import argparse

import time
import torch
from torch import optim, nn

from collections import OrderedDict

from project_utils import load_data, ArchName, create_device

def get_args():
    parser = argparse.ArgumentParser(description='Image Classifier Project Trainer')

    parser.add_argument('data_directory', action="store", help = 'Enter path of data directory')

    parser.add_argument('--save_dir', action="store", dest='save_directory', default = '.', 
                        help = 'Enter path of save directory (Default: .)')
    
    parser.add_argument('--arch', type=ArchName, choices=list(ArchName), default = ArchName.vgg19, help = 'Give the model name (Default: vgg19)')
    parser.add_argument('--learning_rate', action="store", type=float, default = 0.001, help = 'Enter the value of the learning rate (Default: 0.001)')
    parser.add_argument('--hidden_units', action="store", type=int, default = 2048, help = 'Enter the count of nodes in hidden layer (Default: 2048)')
    parser.add_argument('--epochs', action="store", type=int, default = 5, help = 'Enter the count of epochs (Default: 3)')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for training')

    return parser.parse_args()

def create_model(arch_name, hidden_units):
    model = arch_name.load_model()

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(arch_name.get_input_layer_size(), hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.001)),
    
        ('fc2', nn.Linear(hidden_units, 256)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
        
    model.classifier = classifier
    return model

def train_model(device, model, dataloaders, learning_rate, epochs):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(epochs):
        running_loss = 0
        start_time = time.time()
        for inputs, labels in dataloaders['Training']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['Validation']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    test_loss += criterion(logps, labels)

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/len(dataloaders['Training']):.3f}.. "
                f"Validation loss: {test_loss/len(dataloaders['Validation']):.3f}.. "
                f"Validation accuracy: {accuracy/len(dataloaders['Validation']):.3f}.. "
                f"Running time: {time.time() - start_time:.3f}.. seconds")
            model.train()


def save_model(model, save_dir):
    checkpoint = {'arch_name': model.arch_name,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'epochs_count': model.epochs_count,
                  'classifier': model.classifier}
    
    torch.save(checkpoint, save_dir + '/checkpoint.pth')


def main():
    args = get_args()
    arch_name = args.arch
    epochs = args.epochs

    image_datasets, dataloaders = load_data(args.data_directory)

    model = create_model(arch_name, args.hidden_units)
    
    device = create_device(args.gpu)
    train_model(device, model, dataloaders, args.learning_rate, epochs)

    model.cpu()
    model.arch_name = arch_name
    model.class_to_idx = image_datasets['Training'].class_to_idx
    model.epochs_count = epochs

    save_model(model, args.save_directory)

if __name__ == "__main__":
    main()