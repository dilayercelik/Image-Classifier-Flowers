import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models



def parser():
    
    parser = argparse.ArgumentParser(description='To train a neural network')

    parser.add_argument('--data_dir', default = './flowers', help = 'Choose a data directory')
    parser.add_argument('--save_dir', default = '.', help = 'Set directory to save checkpoints')
    parser.add_argument('--gpu', default = 'cpu', help = 'Set GPU for training as "gpu"(recommended), default = "cpu"')
    parser.add_argument('--hidden_units', type = int, default = 4096, help = 'Set the number of hidden units, default = 4096')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Set the learning rate, default = 0.001')
    parser.add_argument('--dropout_rate', type = float, default = 0.2, help = 'Set the dropout probability, default = 0.2')
    parser.add_argument('--epochs', type = int, default = 3, help = 'Set the number of epochs, default = 3')    
    
    args = parser.parse_args()
    
    return args


def load_the_data(data_dir):
    """transform the datasets and build the dataloaders"""
    
    # data directories
    data_dir = 'flowers'
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    
    # transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    
    # load the datasets
    image_datasets = datasets.ImageFolder(data_dir, transform = image_transforms)

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    
    # define the dataloaders
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size = 64, shuffle = True)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)
    
    
    return train_dataset, data_loader, trainloader, validloader, testloader
    
    

def create_model_vgg19(gpu, learning_rate, hidden_units, dropout_rate)
    """build the model from pretrained model 'vgg19' and build its classifier"""
    
    # load pretrained vgg19 model
    model = models.vgg19(pretrained = True)
                       
                       
    # Freeze parameters so we don't backpropagate through them (so features part remains static)
    for param in model.parameters():
        param.requires_grad = False 
    
    
    # Replace VGG-19 classifier with our own classifier
    classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p = args.dropout_rate),
                               nn.Linear(args.hidden_units, 102),
                               nn.LogSoftmax(dim = 1))

    
    model.classifier = classifier   
                       
    # define the loss
    criterion = nn.NLLLoss()

    # define the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    
                       
    # move the model to the available device (recommended = gpu)                   
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu == "gpu" else "cpu")

    model.to(device)
                            
                       
    return model, criterion, optimizer, device                   
    


def train_model(trainloader, validloader, model, epochs, criterion, optimizer, device)
    """training the previously built model in create_model_vgg19()"""
    
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 34
    
    
    
    print("-----Starting Training-----")

    for e in range(epochs):
        for images, labels in trainloader:
            steps += 1  # +1 at each batch
        
            # Move input and label tensors to available device
            images, labels = images.to(device), labels.to(device)

        
            # Training 
            # forward pass
            optimizer.zero_grad()
        
            log_probabilities = model(images)
            loss = criterion(log_probabilities, labels)
        
            # backward pass
            loss.backward()
        
            optimizer.step()  

            running_loss += loss.item()  
        

            # Validation loop
            if steps % print_every == 0:
            
                model.eval()
            
                valid_loss = 0
                valid_accuracy = 0
            
                optimizer.zero_grad()
            
                with torch.no_grad():
                    for images, labels in validloader:
                
                        images, labels = images.to(device), labels.to(device)
                    
                        log_probabilities = model(images)
                        loss = criterion(log_probabilities, labels)
                
                        valid_loss += loss.item() 
        
        
                        # calculate the validation accuracy
                        probabilities = torch.exp(log_probabilities)
                        top_probabilities, top_class = probabilities.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        
                print("Epoch {}/{}..".format(e+1, epochs),
                      "Average Train loss: {:.3f}..".format(running_loss/print_every),
                      "Average Validation loss: {:.3f}..".format(valid_loss/len(validloader)),
                      "Average Validation accuracy: {:.3f}".format(valid_accuracy/len(validloader)))
        
                # end
                running_loss = 0
                model.train()
            

    print("-----Ending Training-----")
    
    
    return model
    
    

def save_checkpoint(model, train_dataset, epochs, optimizer, save_dir)
    """save the model as checkpoint"""
    
    # creating a dictionary 'checkpoint'
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'architecture': 'vgg19',
                  'epochs': args.epochs,
                  'input_size': 25088,
                  'output_size': 102,
                  'features': model.features,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}
    
    # save the checkpoint dictionary
    torch.save(checkpoint, args.save_dir)
    
    return None
    
    

def main():
    
    global args
    args = parse()
    
    
    train_dataset, dataloader, trainloader, validloader, testloader = load_the_data(args.data_dir)
    
    model, criterion, optimizer, device = create_model_vgg19(args.gpu, args.learning_rate, args.hidden_units, args.dropout_rate)
    
    train_model(trainloader, validloader, model, args.epochs, criterion, optimizer, device)
    
    save_checkpoint(model, train_dataset, args.epochs, optimizer, args.save_dir)
            
    print('Finished Training and Saving Checkpoint')   
    
    
    
if __name__ == '__main__':
    main()
       
    
    
    
    
    
    
    
    
    
    
    
                   
    
    
    
    
    

                    
    
