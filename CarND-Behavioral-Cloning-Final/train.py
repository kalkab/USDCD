import csv
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics.tests.test_ranking import test_ranking_appropriate_input_shape
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Activation,MaxPooling2D,Conv2D
from skimage.feature.tests.test_censure import img
from keras.layers.convolutional import Cropping2D
from keras.optimizers import Adam
import matplotlib.image as mpimg
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split

csvFileLoc = 'driving_log.csv' 

def readCSV(csvLoc):
    imgL, angles = [], []
    with open(csvLoc) as csvfile:
        reader = csv.DictReader(csvfile)
        adjustment = 0.2
        for row in reader:
            # print(row['center'], row['steering'])
            strVal = float(row['steering'])
            if (strVal > 0.1 or strVal < -0.1):
                imgL.append(row['center'])
                angles.append(strVal)
                # adding left camera images
                imgL.append(row['left'].strip())
                angles.append(strVal + adjustment)
                # adding right camera images
                imgL.append(row['right'].strip())
                angles.append(strVal - adjustment)
    return imgL, angles

def prepareData(images, steering):
    show_images = False
    x,y = [],[]
    
    availableImages = len(images)
    print("Available number of images:", availableImages)
        
    for i in range(availableImages):
        # take a peek at the images printing out some stats and plotting
        image = mpimg.imread(images[i])
        if (show_images):
            print('Image',images[i],'dimensions:', image.shape,"steering",steering[i])
        #    plt.imshow(image)
        #    plt.show()
        img = cv2.imread(images[i]) # reads BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # back to RGB format
        # adjust brightness with random intensity to simulate driving in different lighting conditions 
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        #print(random_bright)
        img[:,:,2] = img[:,:,2]*random_bright
        img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
       
        flip_prob = np.random.random()
        if flip_prob > 0.5:
            # flip the image and reverse the steering angle
            flipped_image = cv2.flip(img, 1)
            flipped_steering = steering[i]*(-1)
            images[i] = flipped_image
            steering[i] = flipped_steering
            if (show_images):
                print('Flipped image dimensions:', flipped_image.shape,'Steering:',flipped_steering)
                plt.imshow(flipped_image)
                plt.show()
        else :
            images[i] = img
        if i > 40:
            show_images = False
    
    print("Training data:", len(images), ' images')
    
    x = np.array(images)
    #x = np.vstack(images)
    y = np.vstack(steering)
    return x,y

def defModel():
 learning_rate = 0.0001   
 #Based on NVIDIA's End to end convolutional neural network as suggested in Udacity lectures
 model = Sequential()

 model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
 model.add(Cropping2D(cropping =((70,25),(0,0))))
 #===============================================================================
 # starts with five convolutional and maxpooling layers
 #===============================================================================
 model.add(Conv2D(24, (5, 5), padding='same', strides=(2, 2)))
 model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))


 model.add(Conv2D(36, (5, 5), padding='same', strides=(2, 2)))
 model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

 model.add(Conv2D(48, (5, 5), padding='same', strides=(2, 2)))
 
 model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

 model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
 model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

 model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
 model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

 model.add(Flatten())

 # Next, five fully connected layers
 model.add(Dense(1164))
 model.add(Activation('relu'))

 model.add(Dense(100))
 model.add(Activation('relu'))

 model.add(Dense(50))
 model.add(Activation('relu'))

 model.add(Dense(10))
 model.add(Activation('relu'))

 model.add(Dense(1))

 model.summary()

 model.compile(optimizer=Adam(learning_rate), loss="mse", )
 
 return model

#Start with the loading, processing and training based on the images and steering angles

imgLoad, anglesLoad = readCSV(csvFileLoc)
x,y = prepareData(imgLoad, anglesLoad)
modelNvidia = defModel()

Xtr, Xval, Ytr, Yval = train_test_split(x, y, test_size=0.1, random_state=42)
print('Xtr', len(Xtr))
print('Xval',len(Xval))

#===============================================================================
# trainingGen = feed_data_generator(Xtr, Ytr, 64)
# ValidationGen = feed_data_generator(Xval, Yval, 64)
# print(trainingGen.shape)
#===============================================================================
history_object = modelNvidia.fit(x,y,validation_split=0.2, shuffle=True, epochs=7)

#===============================================================================
# history_object = modelNvidia.fit_generator((Xtr,Ytr),
#                  samples_per_epoch = len(Xtr),
#                  validation_data = (Xval,Yval),
#                  nb_val_samples = len(Xval),
#                  nb_epoch = 7,
#                  verbose = 1)
# 
#===============================================================================

## print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#Save model and consequenltly passs to the drive.py
json_string = modelNvidia.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)
modelNvidia.save('model.h5')
print('Model saved')


