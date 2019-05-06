## TODO: define the convolutional neural network architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
# Can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
# import torch._six
# from torch._six import string_classes
import sys
# sys.executable
# sys.path.append('/Users/univesre/miniconda3/envs/cv-nd/lib/python3.6/site-packages/torch')

# 定义并要进行训练的类;
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input, 首先得要是人脸;
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1. input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Four Conv Layers;
        self.conv1 = nn.Conv2d(  1,  32, 5)
        self.conv2 = nn.Conv2d( 32,  64, 3)
        self.conv3 = nn.Conv2d( 64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 2)
        
        # Pooling Layer;
        self.pool = nn.MaxPool2d(2, stride=2)
        
        # Dropout Layer;
#         self.drop1 = nn.Dropout(p = 0.1)
#         self.drop2 = nn.Dropout(p = 0.2)
#         self.drop3 = nn.Dropout(p = 0.3)
#         self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)        
        
        # Full Connect Layer;
        self.fc1 = nn.Linear(36864, 1000)
        self.fc2 = nn.Linear(1000,  1000)
        self.fc3 = nn.Linear(1000,  136)  # 68对点;
        
        # 2. Pooling Layer
        # self.pool1 = nn.MaxPool2d(2, 2)
        # 3. Full Connection Layer;
        # self.fc1 = nn.Linear(32*4, 136)
        ## Note that among the layers to add, consider including:
        # Maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting;
        
    # forward()函数用于定义网络层之间的behavior逻辑;
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model;
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 代码顺序就是层之间的递进顺序;
        # Four Conv Operations;
#         x = self.drop1(self.pool(F.relu(self.conv1(x))))
#         x = self.drop2(self.pool(F.relu(self.conv2(x))))
#         x = self.drop3(self.pool(F.relu(self.conv3(x))))
#         x = self.drop4(self.pool(F.relu(self.conv4(x))))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Dense/flatten the layer;
        x = x.view(x.size(0),-1)  # 36864: 32768 + 4096(=64*64);
        
        # Dropouts;
        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        x = self.fc3(x)  # The final dense;
        
        # Return a modified x, having gone through all the layers of your model;
        return x

