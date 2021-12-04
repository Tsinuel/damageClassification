import torch.nn as nn


class CNNBN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        in_channels = 3 # 3 channels (RGB image)
        out_channels = 64 # hyper param of the conv.
        kernel_size1 = 5
        kernel_size2 = 3
        
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size1,
                               stride = 1,
                               padding=2) # To maintain the same image size
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, 
                                     stride=None,
                                     padding=0)
        
        self.conv2 = nn.Conv2d(in_channels = out_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size2,
                               stride = 1,
                               padding=1)
        
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pool2 = nn.MaxPool2d(kernel_size=2, 
                                     stride=None,
                                     padding=0)

        self.conv3 = nn.Conv2d(in_channels = out_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size2,
                               stride = 1,
                               padding=1)
        
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.pool3 = nn.MaxPool2d(kernel_size=2, 
                                     stride=None,
                                     padding=0)
        self.act = nn.ReLU()
        
        self.sftmax = nn.Softmax(dim=1)
        
        self.flat = nn.Flatten() # flattens all dimensions except batch
        
        self.fc1 = nn.Linear(in_features=16384, out_features=2024)
        self.fc2 = nn.Linear(in_features=2024, out_features=524)
        self.fc3 = nn.Linear(in_features=524, out_features=124)
        self.fc4 = nn.Linear(in_features=124, out_features=4)

    def forward(self, x):
        x = self.pool1(self.bn1(self.act(self.conv1(x))))
        x = self.pool2(self.bn2(self.act(self.conv2(x))))
        x = self.pool3(self.bn3(self.act(self.conv3(x))))
        
        x = self.flat(x)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x)) #if only one-hot encoding used
        #x = self.sftmax(x) #used by the CrossEntropyLoss already
        
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        in_channels = 3 # 3 channels (RGB image)
        out_channels = 64 # hyper param of the conv.
        kernel_size1 = 5
        kernel_size2 = 3
        
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size1,
                               stride = 1,
                               padding=2) # To maintain the same image size
                
        self.pool1 = nn.MaxPool2d(kernel_size=2, 
                                     stride=None,
                                     padding=0)
        
        self.conv2 = nn.Conv2d(in_channels = out_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size2,
                               stride = 1,
                               padding=1)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, 
                                     stride=None,
                                     padding=0)

        self.conv3 = nn.Conv2d(in_channels = out_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size2,
                               stride = 1,
                               padding=1)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, 
                                     stride=None,
                                     padding=0)
        self.act = nn.ReLU()
        
        self.sftmax = nn.Softmax(dim=1)
        
        self.flat = nn.Flatten() # flattens all dimensions except batch
        
        self.fc1 = nn.Linear(in_features=16384, out_features=2024)
        self.fc2 = nn.Linear(in_features=2024, out_features=524)
        self.fc3 = nn.Linear(in_features=524, out_features=124)
        self.fc4 = nn.Linear(in_features=124, out_features=4)

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.pool3(self.act(self.conv3(x)))
        
        x = self.flat(x)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x)) #if only one-hot encoding used
        #x = self.sftmax(x) #used by the CrossEntropyLoss already
        
        return x