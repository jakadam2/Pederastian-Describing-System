import torch.nn as nn
import torch.nn.functional as F
from PAR.cbam import CBAM
import torch

class ColorClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super(ColorClassifier,self).__init__()
        # self.attention_module = CBAM(768)
        self.attention_module = CBAM(2048)
        # self.dl1 = nn.Linear(3072,1024)
        self.dl1 = nn.Linear(8192,1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dl2 = nn.Linear(1024,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dl3 = nn.Linear(512,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dl4 = nn.Linear(128,64)
        self.dl5 = nn.Linear(64,num_classes)
        self.dropout = nn.Dropout(0.3)
        self.avg_pool = nn.AvgPool2d((3,3))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        
        x = self.attention_module(x)
        
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dl1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.dl2(x)
        x = self.bn2(x)
        
        x = self.dropout(x)
        x = self.dl3(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.dl4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dl5(x)
        # x = self.log_softmax(x)
        return x
    


# def color_net(num_classes):
#     # placeholder for input image
#     input_image = Input(shape=(224,224,3))
#     # ============================================= TOP BRANCH ===================================================
#     # first top convolution layer
#     top_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),
#                               input_shape=(224,224,3),activation='relu')(input_image)
#     top_conv1 = BatchNormalization()(top_conv1)
#     top_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_conv1)

#     # second top convolution layer
#     # split feature map by half
#     top_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(top_conv1)
#     top_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(top_conv1)

#     top_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv2)
#     top_top_conv2 = BatchNormalization()(top_top_conv2)
#     top_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv2)

#     top_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv2)
#     top_bot_conv2 = BatchNormalization()(top_bot_conv2)
#     top_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv2)

#     # third top convolution layer
#     # concat 2 feature map
#     top_conv3 = Concatenate()([top_top_conv2,top_bot_conv2])
#     top_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_conv3)

#     # fourth top convolution layer
#     # split feature map by half
#     top_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(top_conv3)
#     top_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(top_conv3)

#     top_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
#     top_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)

#     # fifth top convolution layer
#     top_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
#     top_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv5) 

#     top_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)
#     top_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv5)

#     # ============================================= TOP BOTTOM ===================================================
#     # first bottom convolution layer
#     bottom_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),
#                               input_shape=(227,227,3),activation='relu')(input_image)
#     bottom_conv1 = BatchNormalization()(bottom_conv1)
#     bottom_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_conv1)

#     # second bottom convolution layer
#     # split feature map by half
#     bottom_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(bottom_conv1)
#     bottom_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(bottom_conv1)

#     bottom_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv2)
#     bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)
#     bottom_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv2)

#     bottom_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv2)
#     bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)
#     bottom_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv2)

#     # third bottom convolution layer
#     # concat 2 feature map
#     bottom_conv3 = Concatenate()([bottom_top_conv2,bottom_bot_conv2])
#     bottom_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_conv3)

#     # fourth bottom convolution layer
#     # split feature map by half
#     bottom_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(bottom_conv3)
#     bottom_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(bottom_conv3)

#     bottom_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
#     bottom_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)

#     # fifth bottom convolution layer
#     bottom_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
#     bottom_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv5) 

#     bottom_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)
#     bottom_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv5)

#     # ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================
#     conv_output = Concatenate()([top_top_conv5,top_bot_conv5,bottom_top_conv5,bottom_bot_conv5])

#     # Flatten
#     flatten = Flatten()(conv_output)

#     # Fully-connected layer
#     FC_1 = Dense(units=4096, activation='relu')(flatten)
#     FC_1 = Dropout(0.6)(FC_1)
#     FC_2 = Dense(units=4096, activation='relu')(FC_1)
#     FC_2 = Dropout(0.6)(FC_2)
#     output = Dense(units=num_classes, activation='softmax')(FC_2)
    
#     model = Model(inputs=input_image,outputs=output)
#     sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#     # sgd = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
#     model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return model