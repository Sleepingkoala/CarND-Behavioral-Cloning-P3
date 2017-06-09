# **Udacity SDCNanoDegree P3: Behavioral Cloning ** 

## Writeup Template

### Here I use the writeup file as a markdown file, as usual. 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior. Here I used the dataset provided in project resources.And I didn't record training data by myself.
* Build a convolution neural network with Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that trained model drives around track one at least one lap successfully. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a well-written report


[//]: # (Image References)

[image1]: ./examples/NVIDIA_CNN_Architecture.jpg "Model Visualization"
[image2]: ./examples/left_example1.jpg "left1"
[image3]: ./examples/center_example1.jpg "center1"
[image4]: ./examples/right_example1.jpg "right1"
[image5]: ./examples/left_example2.jpg "left2"
[image6]: ./examples/center_example2.jpg "center2"
[image7]: ./examples/right_example2.jpg "right2"
[image8]: ./examples/left_example3.jpg "left3"
[image9]: ./examples/center_example3.jpg "center3"
[image10]: ./examples/right_example3.jpg "right3"

[image11]: ./examples/original_left.jpg "original left"
[image12]: ./examples/cropped_image.jpg "cropped Image"
[image13]: ./examples/resized_image.jpg "resized Image"
[image14]: ./examples/sheared_image.jpg "sheared image"
[image15]: ./examples/flipped_image.jpg "flipped image"
[image16]: ./examples/adjusted_image.jpg "adjusted image"
[image17]: ./examples/data_pipeline.jpg "data processing pipeline"

[image18]: ./examples/performances/CNNwithDropout_1.png "final architecture1"
[image19]: ./examples/performances/CNNwithDropout_2.png "final architecture2"

[image20]: ./examples/performances/NoDropout_epoch8.png "performance1"
[image21]: ./examples/performances/epoch4.png "performance2"
[image22]: ./examples/performances/epoch8.png "performance3"
[image23]: ./examples/performances/epoch16.png "performance4"
[image24]: ./examples/performances/epoch32_1.png "performance5"
[image25]: ./examples/performances/epoch32_2.png "performance6"

[image26]: ./examples/performances/20_47_49.png "snapshot1"
[image27]: ./examples/performances/20_48_40.png "snapshot2"
[image28]: ./examples/performances/20_48_57.png "snapshot3"
[image29]: ./examples/performances/20_49_12.png "snapshot4"
[image30]: ./examples/performances/20_49_31.png "snapshot5"
[image31]: ./examples/performances/20_50_09.png "snapshot6"
[image32]: ./examples/performances/20_50_25.png "snapshot7"
[image33]: ./examples/performances/20_50_48.png "snapshot8"
[image34]: ./examples/performances/20_51_14.png "snapshot9"
[image35]: ./examples/performances/20_51_27.png "snapshot10"
[image36]: ./examples/performances/20_51_54.png "snapshot11"
[image37]: ./examples/performances/20_52_30.png "snapshot12"
[image38]: ./examples/performances/20_52_43.png "snapshot13"


---

## Rubric Points
### Here I will refer to the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained keras convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 recording the vehicle driving autonomously around the track for at least one full lap

Besides, the helper.py and model.json files are also included. The helper.py includes my self built-in functions about data loading, preprocessing, augmentation, and model saving. The model.json is created with model.h5 at the same time when model.py is excuted. 

#### 2. Submission includes functional code
My platforms are a local lapbook with Ubuntu 14.04 LTS and an AWS instance. I designed and trained my CNN architecture on AWS with model.py and helper.py, then test it with simulator on local terminal. Later the model.h5 and model.json are created on AWS by executing
```sh
python model.py
```

The model.h5 and model.json are downloaded from AWS to local lapbook. Before testing the car in autonomous mode in simulated environment, the simulator should be opened in terminal by:
```sh
cd ./CarND-Behavioral-Cloning-P3/beta_simulator_linux
./ beta_simulator.x86_64
```
And then using the Udacity provided simulator Unity and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Once the auronomous vehicle drives successfully in track one, the video of results counld be saved by executing
```sh
python drive.py model.h5 run1
```
Then the images in filefolder run1 are named by the timestamps of when the images were seen. This information is used by video.py to create a chronological video of the vehicle driving by executing
```sh
python video.py run1
```
Particularly, I used the python generator to generate training data instead of storing them in memory, by executing model.py codeline 70-71:
```sh
train_gen = helper.generate_next_batch()
validation_gen = helper.generate_next_batch()
```

#### 3. Submission code is usable and readable

The model.py and helper.py files contain the code for designing, training and saving the convolution neural network. For details, helper.py includes my self built-in functions about data loading, preprocessing, augmentation, and model saving. Model.py contains the scripts to design and train the model, respectively. 

And the drive.py includes my preprocessing produres for testing data and communicating with simulator. The video.py records the test frames as .mp4 for performance reviews. The following shows the pipeline I designed for training and validating the model, and it contains my notes to explain how the code works.

---

### Overview of Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is inspired on NVIDIA's CNN Architecture "End to End Learning for Self-Driving Cars" Arxiv2016.(Source:[NVIDIA CNN Arxiv2016](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf))

![alt text][image1]

My model consists of a data normalization layer using a Keras lambda layer(model.py code line 21),five convolutional layers with different filter sizes from 5x5 to 3x3 and depths between 24 and 64 channels, followed by RELU Activation and max-pooling layers (model.py code lines 25-43). Then this architecture includes a flatten layer(model.py code line 45), five fully-connected layers followed by RELU activation layers to introduce nonlinearity (model.py code line 49-63).

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers with keep_prob 0.5, in order to reduce overfitting, according to my experience in former project 2 and the training for modified MSCNN (model.py conde lines 51 and 55). 

Besides, the training data and validation data are generated by different python generators with the split ratio 3.13 empirically, to ensure that the model was not overfitting (model.py code line 70-71, and helper.py codeline 172-193). 

Finally, I used two traditional data preprocessing functions and three data augmentation functions to improve the generalization power of this model, as described later. After training, my model was tested on track 1 with autonomous mode on the simulator 'beta_simulator_linux' . Accordingt to the result video.mp4, this model ensures that the vehicle always stays on the track during the whole testing process. This proves these attempts are useful for overfitting reduction.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py code line 67). And there are other hyperparameters finally chosen like:

* activation function: RELU
* number of epochs: 16
* batch_size: 64
* dropout keep_prob: 0.5

I tuned different parameter settings with lots of trials and errors. Here I used a debugging solution which is a combination of a well-known architecture and an iterative approach.  

#### 4. Appropriate training data

Since I chose the training data provided by this course resources. After the exploration and visualization of the dataset, to keep the vehicle driving on the road, I used a combination of center lane driving, recovering from the left/right sides of the road. Besides, I used two traditional data preprocessing functions and three data augmentation functions to enhance the variety of traning data. And the result has proven the generated training data is appropriate.

For details about how I created the training data, here is a pipeline of the training data processing. The three optional functions in dashline box are random_shear(warping),random_horizontally_flipping, and intensity_adjustment for data augmentation. And two necessary functions in solid line box are crop()and resize()for traditional data preprocessing, respectively. You can see the details as follows.

![alt text][image17]

---

### Details about Model Architecture and Training Strategy

#### 1. Solution Design Approach

My model is a convolution neural network model similar to the inspiration of NVIDIA's CNN Architecture "End to End Learning for Self-Driving Cars" 2016.(Source:[NVIDIA CNN Arxiv2016](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)). I thought this model might be appropriate because they handled similar image and steering angle data and had similar running process.

**However, there are three main differences between my model and NVIDIA's model**:

* the input image size is cropped and resized as 64x64x3, taking the most useful portion of image for learning and prediction, where input image is 66x200x3 in NVIDIA's model, bringing lots of computation burden.

* the max-pooling layer is applied just after each convolutional layer to reduce the calculation burden,i.e. training time, where the NVIDIA's model has none.

* two dropout layers are used after two fully-connected layers, to prevent overfitting, while NVIDIA's model didn't. And the result has proven this is right choice.


The details about my CNN architecture can be seen later, by executing model.py code line65:
```sh
model.summary()
```
Fortunately, my first model had a low mean squared error on the training set and validation set as follows. This implied that the model was converged and not overfitting. My solution of design approach is right.

![alt text][image20]

When running the simulator to see how well the car was driving around track one, we can find there were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I used the recovery left/right training data and augmented data. At the end of the process, the vehicle can drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric). 

You can see my model consists of a data normalization layer using a Keras lambda layer(model.py code line 21),five convolutional layers with different filter sizes from 5x5 to 3x3 and depths between 24 and 64 channels, followed by RELU Activation and max-pooling layers (model.py code lines 25-43). Then this architecture includes a flatten layer(model.py code line 45), five fully-connected layers followed by RELU activation layers to introduce nonlinearity, and two dropput layers to reduce the overfitting(model.py code line 49-63).

![alt text][image18]
![alt text][image19]

#### 3. Exploration and Preparation of the Training and validation dataset

##### Dataset statistics and visualization
Here I used the dataset provided by course resources about 24,108 images consisting of 8036 center-view and related 8036 left/8036 right-view images.And I didn't record images by myself. The dataset includes JPG images with 320x160x3 dimensions. As the NVIDIA's paper said,the training images were sampled at 10 FPS from videos, because a higher sampling rate would result in images that are highly similar and thus not provide much useful information. Here are some examples from the dataset:

 Left|  Center| Right|
-----|--------|-------
![left][image2] | ![center][image3] | ![right][image4]
![left][image5] | ![center][image6] | ![right][image7]
![left][image8] | ![center][image9] | ![right][image10]

The training track contains a lot of shallow turns and straight road segments. Therefore the steering angles are mostly 0. Then preprocessed data is necessary for unknown validation scenes before putting in the trained CNN as described in data processing pipeline above. 

##### Data preprocessing and data augmentation

In the original NVIDIA's paper, the authors augment the data by adding artificial shifts and rotations to teach the nerwork how to recover from a poor position or orientation. The magnitude of these pertubations is chosen randomly from a normal distribution. In this project, my data preprocessing includes two parts: necessary traditional preprocessing and optional augmentation preprocessing. 

**The traditional processing includes 2 functions(necessary):**

* cropping. Through the examples of the training dataset, I notice not all of the pixels contain useful information. For example, in the original example image, the top portion of the image captures trees and sky, and the bottom protion of the image captures the hood of the car. Besides, cropping the top and bottom parts will help train the classifier faster if unuseful pixels are removed and the calculation burden is reduced. Here is a pair of examples for an original input image and its cropped version as follows:

original input|  Cropped|
-----|--------|-------
![left][image11] | ![right][image12]

* resize. Similarly, to reduce training time, the cropped image could be resized to 64x64 and fed into my CNN architecture. The following figures show the result of this operation applied to cropped image.

Cropped| Resized|
-------|--------|-------
![left][image12] | ![right][image13]

**Data Augementation(optional)**

The augmentation for training dataset includes functions as follows:

* random shear. Also known as warping. Applies a random 2D affine transformation to images in a shear range(-200, 201). Here I chose 90% of original images and steering angles randomly to do this operation. And the rest 10% are kept to help car navigation in training track 1. The following figures show the result of this operation to an original image.

original input|  sheared|
-----|--------|-------
![left][image11] | ![right][image14]

* random horizontal flip. Because the images and steering angles are learned and recognized semantically, not like the traffic signs in project 2. So flipping operation would not influence the testing accuracy. Here I randomly chose 50% of the sheared images to flip horizontally.The idea behind this operation is left turning bends are more prevalent than right bends in the training track.Hence to increase the generalization of my model, I fliped images and respective steering angles. The following figures show the result of this operation to a sheared image.


sheared| ![left][image14]|
-----|--------|-------
fliped| ![right][image15]


* random intensity adjustment. Adjusting the intensities of images in new scales. Firstly converting the RGB images into HSV colorspace, then adding random uniform variations to the V channel. And converting the changed image into RGB colospace again. This operation simulates the car driving in different illumanation conditions. The following figures show the result of this operation to a flipped image.

flipped | ![left][image15]|
--------|--------|-------
adjusted| ![right][image16]

The reference docs:

* [opencv Geometric Image Transformation](http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=warpaffine)
* [cnblogs](http://www.cnblogs.com/pakfahome/p/3914318.html)

Data is augmented and generated on the fly using python generators. So for every epoch, the optimizer practically sees a new and augmented data set. 

####  4. Training Process

After traditional data preprocessing and data augmentation, the training dataset was very large and it couldn't be stored in memory. Hence Keras' fit_generator method is used to train images generated by the generator.

In this project two genrators are created:

* train_gen = helper.generate_next_batch()
* validation_gen = helper.generate_next_batch()

And I used 20032 images(313 batches) per training epoch. These images are generated using the data processing pipeline described above. Besides, 6400 images (also generated on the fly, 100 batches)were used for each validation epoch. Here is the final configuration of my model for training and validation: 

* Optimizer: Adam Optimizer.
* Training set: 20,032 imges, generated on the fly
* Validation set: 6,400 images, generated on the fly
* number of epochs: 16.
* batch_size for training set and validation set: 64.
* learning_rate: 0.0001.I used an adam optimizer so that manually training the learning rate wasn't necessary.
* dropout keep_prob: 0.5.

Note that no testing set is used, since the success of the model is evaluated by how well it drives on the road and not by test set loss.

Finally, when it comes to the number of training/validation epochs I tried several configuration such as 4, 8, 16, 32. Considering the tradeoff between training/validation loss and computation time(training takes a long time even on AWS GPUs), epochs = 16 gives the best balance on both training and validation tracks. Here are some performances of my model with different epochs.

epochs = 4  | ![center][image21]|
------------|--------------------
epochs = 8  | ![center][image22]|
epochs = 16 | ![center][image23]|
epochs = 32 | ![center][image24]|
epochs = 32 | ![center][image25]|


###  Simulation Results

From the result video.mp4 we can see that the tires of the car never leaves the drivable part of the track1 surface. And it also didn't pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle). You can find the video.mp4 in this repository.

#### Validation track

There are 13 snapshots of the car driving during the whole track1, as follows:

![alt text][image26]
![alt text][image27]
![alt text][image28]
![alt text][image29]
![alt text][image30]
![alt text][image31]
![alt text][image32]
![alt text][image33]
![alt text][image34]
![alt text][image35]
![alt text][image36]
![alt text][image37]
![alt text][image38]
