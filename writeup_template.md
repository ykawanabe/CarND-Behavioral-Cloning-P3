# Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_2017_07_15_18_25_31_621.jpg "Center driving"
[image3]: ./examples/center_2017_07_15_18_26_33_753.jpg "Sharp curve"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


---
### Files Submitted & Code Quality

#### Files
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp for autonomous mode driving example with the learned model

#### How to run the model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model first preprocess the input images. Crop the top and bottom of the images where a hood and landscape are always captured. After that, the image is resized into 64x64 in order to reduce the calculation time by reducing the input size.

The preprocessed images will then goes into the neural network model. My model is based on [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). It consists of convolution layers, fully connected layers and dropout layers. The convolution layers is consisted of three 5x5 and two 3x3 layers and depths between 24 and 64 (model.py lines 94-98)

The Activation function for the convolution layers is RELU  to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 100, 103 and 105).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 74-76). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 109).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, sharp curve and transition to bridges.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to capture essential feature values from images.

My first step was to use a convolution neural network model similar to the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) I thought this model might be appropriate because this model is specifically designed for autonomous driving and has multiple deep convolution layers which can capture complicated feature values from the images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding dropout layers so that the model can be more generic.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. For example, the model did not handle entering the bridge and sharp curves well. To improve the driving behavior in these cases, I added more data for those cases and reduced training data where lane is straight. In addition to that, I also increased steering correction value for left and right images of the car, and removed the images where steering angles are lower than certain angle to give the model more data to train for sharp curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 90-107) consisted of a convolution neural network with the following layers:

* Three 5x5 convolution layer with depths between 24 and 64
* Two 3x3 convolution layer with depth 64
* Flatten layer
* Five fully connected layer starting with 1164 neurons with dropout layers
* Steering angle output

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle making a sharp curve so that the vehicle would learn how to make a curve by looking at the image in front of the vehicle. These images show what a sharp curve looks like:

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
