##############################################################################
"""
File name: train_classifier.py
Witten by: Abiy Melaku, Aaron John and Tsinuel Geleta
Date: Dec 3, 2021


This code implements a training, validation and testing algorithm of the CNN
model for building damage classification. The code for the CNN model is 
separately written in the 'model.py' file at the end. 

"""
##############################################################################
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import models
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Configurations
NUM_CLASSES = 4
IMAGE_SIZE = 128
BATCH_SIZE = 64
RANDOM_SEED = 123

damage_levels = dict()
damage_levels[0] = 'no-damage' 
damage_levels[1] = 'minor-damage' 
damage_levels[2] = 'major-damage' 
damage_levels[3] = 'destroyed'

print('Batch size: ', BATCH_SIZE)
print('Number of classes: ', NUM_CLASSES)
print('Image size: ', IMAGE_SIZE)
print('Random size: ', RANDOM_SEED)

start_time = time.time()


#Load and prepare the training and validation data
def prepare_data(train_path, valid_path, test_path):
    
    #Data augmentation
    transform = transforms.Compose([transforms.RandomResizedCrop
                                    (size=IMAGE_SIZE, scale=(0.8, 1.0)),
                                    transforms.RandomRotation(degrees=15),
                                    transforms.ColorJitter(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_path, transform=transform)
    
    
    print("Train set: {}, Valid set: {}\n".format(len(train_dataset), 
                                                    len(valid_dataset)))
    

    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True)


    
    #Only resize the image data
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                transforms.ToTensor()])


    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    
    test_size = len(test_dataset)
    
    print("Testing data set: {}\n".format(test_size))
    
        
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=True)

    
    return train_loader, valid_loader, test_loader


def compute_class_weight(device, data):
    
    class_counts = dict(Counter(data.dataset.targets))
    total_count = len(data.dataset)
    
    class_weights = torch.zeros(NUM_CLASSES, device=device) 

    for i in range(NUM_CLASSES):
        class_weights[i] = float(class_counts[i])/total_count
    
    return class_weights

def plot_sample_data(data):
    data_enum = enumerate(data, 101)
    batch_idx, (images, labels) = next(data_enum)  
    
    fig = plt.figure(figsize=(15, 9))
    for i in range(15):
        plt.subplot(3,5,i+1)
        plt.tight_layout()
        plt.imshow(np.transpose(images[i]))
        plt.title("Damage level: {}".format(damage_levels[labels[i].item()]))
        plt.title(damage_levels[labels[i].item()], fontsize=20)
    plt.show()

# Function for training and validation of the the model
def train_model(device, model, train_data, valid_data, num_epochs, optimizer, 
                criterion):
    
    # ====== Data collection ====== #
    epochs = [] 
    train_loss, train_f1, train_acc =  [], [], []
    valid_loss, valid_f1, valid_acc = [], [], []

    # ====== Loop ====== #
    for epoch in range(num_epochs):  
        # ====== Train ====== #
        model.train() # Set the model be 'train mode' 

        acc_sum = 0.0
        loss_sum = 0.0
        f1_sum = 0.0
        count = 0
        
        #Iterate over each batch
        for i, data in enumerate(train_data, 0):
            
            # get the inputs and labels [inputs, labels]
            # inputs, targets = data
            inputs, targets = data[0].to(device), data[1].to(device)
            
            # Initialize the gradient in the optimizer
            optimizer.zero_grad() 

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            acc_sum += balanced_accuracy_score(targets.cpu(), predicted.cpu())
            f1_sum += f1_score(targets.cpu(), predicted.cpu(), 
                               average='weighted')
            count += 1

        epochs.append(epoch)
        train_loss.append(loss_sum/count)
        train_acc.append(acc_sum/count)
        train_f1.append(f1_sum/count)

        # ====== Validation ====== #
        model.eval() # Set the model be in 'evaluation mode' 
        acc_sum = 0.0
        loss_sum = 0.0
        f1_sum = 0.0
        count = 0

        with torch.no_grad():
            for i, data in enumerate(valid_data, 0):
                inputs, targets = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # the class with the highest energy is the prediction
                _, predicted = torch.max(outputs.data, 1)
                
                loss_sum += loss.item()
                acc_sum += balanced_accuracy_score(targets.cpu(), 
                                                   predicted.cpu())
                f1_sum += f1_score(targets.cpu(), predicted.cpu(), 
                                   average='weighted')
                count += 1

        valid_loss.append(loss_sum/count)
        valid_acc.append(acc_sum/count)
        valid_f1.append(f1_sum/count)

        end_time = time.time()        
        
        print('Epoch: {}/{}, Train Loss: {:.6f}, Train Acc: {:.3f}%, Val \
              Loss: {:.6f}, Val Acc: {:.3f}%, Val F1: {:.3f}, \
                  Time: {:.0f} sec:'.format
              (epoch + 1, num_epochs, train_loss[-1], train_acc[-1]*100, 
               valid_loss[-1], valid_acc[-1]*100, valid_f1[-1], 
               end_time - start_time))

    return epochs, train_loss, train_acc, train_f1, \
        valid_loss, valid_acc, valid_f1

def test_model(device, model, test_data):

    model.eval() # Set the model be in 'evaluation mode'   
    count = 0
 
    test_class_acc = np.zeros(NUM_CLASSES)
    test_f1 = np.zeros((NUM_CLASSES, 4))
    test_tot_acc = np.zeros(1)
    test_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    correct_pred = {classname: 0 for classname in damage_levels}
    total_pred = {classname: 0 for classname in damage_levels}

    with torch.no_grad():
        for i, data in enumerate(test_data, 0):
            
            inputs, targets = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
                        
            # A class with the highest energy is the prediction
            _, predictions = torch.max(outputs.data, 1)
            
            f1 = precision_recall_fscore_support(targets.cpu(), 
                                                 predictions.cpu(), 
                                                 average=None,
                                                 labels=[0,1,2,3],
                                                 zero_division=0.0)
            
            # Collect the correct predictions for each class
            for label, prediction in zip(targets, predictions):
                if label == prediction:
                    correct_pred[label.item()] += 1
                total_pred[label.item()] += 1    

            count += 1
            test_tot_acc[0] += balanced_accuracy_score(targets.cpu(),
                                                       predictions.cpu())
            test_conf_matrix += confusion_matrix(targets.cpu(), 
                                                 predictions.cpu(), 
                                                 labels=[0, 1, 2, 3])
                
            for i in range(NUM_CLASSES):
                test_f1[i,0] += f1[0][i]
                test_f1[i,1] += f1[1][i]
                test_f1[i,2] += f1[2][i]
                test_f1[i,3] += f1[3][i]

    test_f1/=count
    test_tot_acc/=count
    test_conf_matrix/=count
    
    # Calculate the accuracy for each class
    for level, count in correct_pred.items():
        test_class_acc[level] = float(count)/total_pred[level]
        print("Accuracy for {:5s} is: {:.2f} %".format(
            damage_levels[level], 100*test_class_acc[level]))
    
    print("Final balanced accuracy: {:.2f} %".format(100*test_tot_acc[0]))

    return test_tot_acc, test_class_acc, test_f1, test_conf_matrix

# Create, Train and Evaluate and Test the model
def main():    
    
    #Set up device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(RANDOM_SEED)
    
    train_path = "../data/train"
    valid_path = "../data/validation"
    test_path = "../data/test"
    
    train_data, valid_data, test_data = prepare_data(train_path, 
                                                     valid_path, 
                                                     test_path)
    
    plot_sample_data(train_data)
    
    model_type = 'CNNBN'

    optimizer_types = ['sgd','adam']
    learning_rates = [0.00001, 0.00005, 0.0001]
    num_epochs = [10, 25, 50, 100]
    momentum = 0.9

    print('Train class distribution', 
          dict(Counter(train_data.dataset.targets)))    
    print('Validation class distribution',
          dict(Counter(valid_data.dataset.targets)))    
    print('Test class distribution', 
          dict(Counter(test_data.dataset.targets)))    


    print('\n================ Hyper parameters to be tested ================')
    print('Optimizers: ', optimizer_types)
    print('Learning rates: ', learning_rates)
    print('Number epochs: ', num_epochs)
    
    print('\n================ Training Running! ================')
    print('Model type: ', model_type)

    for opt in range(len(optimizer_types)):

        for lr in range(len(learning_rates)):
            for epch in range(len(num_epochs)):

                if model_type=='CNNBN':
                    model = models.CNNBN()
                elif model_type=='CNN':
                    model = models.CNN()

                    
                if torch.cuda.device_count() > 1:
                    print("Using ", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)

                model.to(device)
                
                class_weights = compute_class_weight(device, train_data)    
                criterion = nn.CrossEntropyLoss(weight=class_weights)

     
                assert optimizer_types[opt] in ("adam", "sgd")
                
                print('Optimizer type size: ', optimizer_types[opt])
                print('Number of epochs: ',  num_epochs[epch])
                print('Learning rate: ', learning_rates[lr])
                
                if optimizer_types[opt] == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), 
                                                 lr=learning_rates[lr], 
                                                 betas=(0.9, 0.999))
                
                if optimizer_types[opt] == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), 
                                                 lr=learning_rates[lr], 
                                                 momentum=momentum)
                    
                epochs, train_loss, train_acc, train_f1, valid_loss, \
                    valid_acc, valid_f1 = train_model(device, model,
                                                      train_data, 
                                                      valid_data, 
                                                      num_epochs[epch], 
                                                      optimizer, 
                                                      criterion)
                
                filename = 'MDL_{}_OPT_{}_LR_{:.1e}_EP_{}'.format(
                    model_type, optimizer_types[opt], 
                    learning_rates[lr], num_epochs[epch])
                          
                print('Writing to file: ', filename)

                out_path = 'output/'
                
                np.savetxt(out_path + 'epochs_' + filename, 
                           np.asarray(epochs))
                np.savetxt(out_path + 'train_loss_' + filename, 
                           np.asarray(train_loss))
                np.savetxt(out_path + 'train_acc_' + filename, 
                           np.asarray(train_acc))
                np.savetxt(out_path + 'train_f1_' + filename, 
                           np.asarray(train_f1))
                
                np.savetxt(out_path + 'valid_loss_' + filename, 
                           np.asarray(valid_loss))
                np.savetxt(out_path + 'valid_acc_' + filename, 
                           np.asarray(valid_acc))
                np.savetxt(out_path + 'valid_f1_' + filename, 
                           np.asarray(valid_f1))
                
                print('\n================ Testing Running! ================')

                test_tot_acc, test_class_acc, test_f1, test_conf_matrix = \
                    test_model(device, model, test_data)
                
                np.savetxt(out_path + 'test_tot_acc_' + filename, 
                           test_tot_acc)
                np.savetxt(out_path + 'test_class_acc_' + filename, 
                           test_class_acc)
                np.savetxt(out_path + 'test_f1_' + filename, test_f1)
                np.savetxt(out_path + 'test_conf_matrix_' + filename, 
                           test_conf_matrix)

if __name__ == '__main__':
    main()
