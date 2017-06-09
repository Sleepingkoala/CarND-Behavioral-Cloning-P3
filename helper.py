import os
import json
import errno

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.stats import bernoulli

# training data directionary and steering_bias
driving_log_file = './data/driving_log.csv'
image_path = './data/'
steering_coefficient = 0.25

# for data augmentation: random_shear, random_horizontally_flip, intensity_adjustment
def random_shear(image, steering_angle, shear_range = 200):
    """
    shear an image between a shear interval with given parameters.
    
    parameters: image: input image
                steering_angle: the steering angle of the image
                shear_range: random shear between [-shear_range,shear_range +1] will be implemented.
                
    return: the randomly sheared image    
    """
    n_rows,n_cols,n_channels = image.shape
    dx = np.random.randint(-shear_range, shear_range+1)
    
    random_point = [n_cols/2 + dx, n_rows/2]
    pts1 = np.float32([[0,n_rows],[n_cols,n_rows],[n_cols/2,n_rows/2]])
    pts2 = np.float32([[0,n_rows],[n_cols,n_rows],random_point])
    
    dsteering = dx/(n_rows/2) *360/(2*np.pi*25.0)/6.0
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(n_cols,n_rows),borderMode = 1)
    steering_angle += dsteering
    
    return image,steering_angle

def random_horizontally_flip(image,steering_angle,flipping_prob = 0.5):
    """
    flipping 50% of the original images to reduce the turning left bias preseted in training data. 
    And if  the image is flipped, the steering angel will be nagated.
    
    parameters: image: input image
                steering_angle: the applied steering angle
                
    return both flipped image and new steering angle.
    """
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1*steering_angle
    else:
        return image,steering_angle
    
def intensity_adjustment(image):
    
    """
    adjust the input image with random intensity variation.
    
    return the adjusted image.
    """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    random_intensity = 0.25+ np.random.uniform()
    image1[:,:,2] = image1[:,:,2]* random_intensity
    
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


# for classic preprocessing: crop() and resize()

def crop(image,top_crop,bottom_crop):
    """
    crop an image from both top and bottom using given parameters.
    
    parameters: image: input image
                top_crop: the pixels will be cropped from the top of input image
                bottom_crop: the pixels will be cropped from the bottom of the input image.
                
    return: the cropped image    
    """
    top = top_crop
    bottom = image.shape[0] - bottom_crop
    cropped_image = image[top:bottom,:,:]
    
    return cropped_image

def resize(image,target_size):
    """
    resize an image into target_size.
    
    parameters: image: input image
                target_size: a tuple representing the resize dimension
                
    return: the resized image    
    """
    return cv2.resize(image,target_size)

def preprocess_image(image,top_crop = 55,bottom_crop = 25,target_size = (64,64)):
    """
    crop and resize the input images, and then normalize the training data.
    
    return normalized image
    """
    image = crop(image,top_crop,bottom_crop)
    image = resize(image,target_size)
    return image

# for data augmentation
def generate_new_image(image, steering_angle,top_crop = 55, bottom_crop = 25,do_shear_prob = 0.9):
    """
    implement data augmentation with above helper funcitons.
    
    parameters:image: input image
               steering_angle: steering_angle
               top_crop,bottom_crop: cropped pixels in function crop()
               do_shear_prob: the probability of doing random_shear()
               
    return the augmented image and steering_angle.
    """    
    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image,steering_angle = random_shear(image,steering_angle) 
    
    # random horizontally flipping
    image,steering_angle = random_horizontally_flip(image,steering_angle)
    
    # implement intensity adjustment
    image = intensity_adjustment(image)
    
    # similar preprocess the image(crop,resize and normalize)
    image = preprocess_image(image,top_crop = 55,bottom_crop = 25,target_size = (64,64))
    return image, steering_angle

def get_next_image_files(batch_size = 64):
    """
    the simulator records left,center and right images at a given time. However, when I pick images for training,
    I randomly pick one of the three images and its steering angle.
    
    return a list of selected image filenames and respective steering angles.
    """
    data = pd.read_csv(driving_log_file)
    num_of_images = len(data)
    random_indices = np.random.randint(0,num_of_images,batch_size)
    
    image_files_and_angles = []
    
    for index in random_indices:
        random_image = np.random.randint(0,3)
        
        if random_image == 0:
            image = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering']+ steering_coefficient
            image_files_and_angles.append((image,angle))
        
        elif random_image == 1:
            image = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((image,angle))
        
        else:
            image = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering']-steering_coefficient
            image_files_and_angles.append((image,angle))
            
    return image_files_and_angles

def generate_next_batch(batch_size = 64):
    """
    the generator yields the next training batch.
    
    yield/return a tuple of features and steering angles as two numpy arrays.
    """
    while True:
        X_batch = []
        y_batch = []
        image_files_and_angles = get_next_image_files(batch_size)

        for image_file,angle in image_files_and_angles:
            raw_image = plt.imread(image_path+image_files)
            raw_angle = angle
            
            new_image,new_angle = generate_new_image(raw_image,raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)
            
        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be true'
        
        yield np.array(X_batch),np.array(y_batch)
            

def save_model(model, model_name = 'model.json', weights_name = 'model.h5'):
    """
    save the model into disk.
    
    parameters: model: keras model to be saved.
                model_name: the name of the model file
                weights_name: the name of the weight file
                
    return none.
    """
    silent_delete(model_name)
    silent_delete(weights_name)
    
    json_string = model.to_json()
    
    with open(model_name,'w') as outfile:
        json.dump(json_string,outfile)
        
    model.save(weights_name)
    
def silent_delete(file):
    """
    remove the given file from the file system if it`s available.
    
    parameters: file: file to be removed.
    
    return none.
    """
    try: 
        os.remove(file)
    
    except OSERROR as error:
        if error.errno != errno.ENOENT:
            raise
    