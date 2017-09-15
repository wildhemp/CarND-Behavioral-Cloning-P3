#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 64 and fully connected layers with feature sizes between 50-1000 (model.py lines 255-292)

The model includes RELU layers to introduce nonlinearity, and the data is normalized and cropped in the model using a Keras lambda and cropping layers (model.py lines 262-264).

The model is a somewhat modified version of the nvidia model, tuned to work better for the current problem.

####2. Attempts to reduce overfitting in the model

The model uses L2 regularization to prevent overfitting (e.g. model.py line 266). I also experimented with using Dropout layers, which worked, but using L2 made the loss decrease smoother (not jumping up and down), and the car run smoother on Track 1.

The model was trained and validated on different data sets (both track 1 and 2) to ensure that the model was not overfitting (code line 298-302).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for both Track 1 and Track 2.

I also modified the speed in drive.py to use the maximum, to see how well the model is able to keep up. It still drives both tracks well, although on Track 2 it gets very close to running off the road a couple times, but manages to (barely) stay on the the road.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 290).

The L2 regularization is set to 0.001, which works quite well in preventing overfitting.

The final model was trained for 5 epochs, which seemed to lead to the best result when testing on both tracks 1 and 2.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the udacity provided data, and recorded data from track 2 as well. For recovery the model was using the left and right images as well as horizontal translations (only necessary for the challange track).

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a somewhat simple model and modify it so that overfitting to a small data it results the lowest training loss possible then change it to prevent heavy overfitting.

My first model consisted of 2 convolutional layers of 32 and 64 feature maps and 2 max pooling layer between them, all using strides of 2. This was followed by 3 fully connected layers of 400-100-1 features. This model has some similarity to LeNet, but uses larger feature maps for the convolutional layer to compensate for the larger input size.

From this point I started adding/removing layers and changing feature maps/sizes so that the model results in lower training loss. I trained the model on a smaller dataset until the training loss started to increase again or just imply didn't decrease anymore, then repeated the process.

I tried a couple different combinations and quickly arrived to a model very similar to the nvidia model. In the process I also got rid of the max pooling layers, because they didn't add any value anymore. Using strides of 2 on every layer reduced the size of the dataset anyways.

The main difference in my model vs the nvidia model is:

1. It starts 32 feature maps vs nvidia's 24. I tried 24, but it made things worth.
2. It has larger fully connected layers, mainly to compensate the data size coming out of the convolutional layers.

To combat the overfitting, I tried 2 strategies:

1. Using dropout layers
2. L2 regularization

Using dropout layers worked quite well, but it made the training/validation loss jump around quite a lot, making it hard to judge where did the model start to overfit vs. it an still learn something. Also, the car seemed to be driving a little more erratically on track 1.

So I gave a try to the L2 regularization and after tuning the data a little bit, it wokred very well. At the end I decided to use this instead of dropouts, because it was easier to figure out how many epochs to train and the car seemed to drive a little smoother.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the model either drove off the road or got very close to it. To combat this the model was augmented with the left and right camera images and filtered, so that includes fewer data points with angle of 0.

Also for track 2 additional augmentation was needed. Doing random horizontal translations made the model much better in staying on the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 255-292) consisted of a convolution neural network with the following layers and layer sizes:


|Layer                          | Description                                     |
|:-----------------------------:|:-----------------------------------------------:|
| Input                         | 160x320x3                                       |
| Cropping2D                    | 50x320x3                                        |
| Normalization (Lambda)        | 50x320x3                                        |
| Convolution2D                 | 25x160x32 - 'same' padding, 2x2 strides, RELU, L2 0.001 |
| Convolution2D                 | 13x80x48 - 'same' padding, 2x2 strides, RELU, L2 0.001 |
| Convolution2D                 | 7x40x64 - 'same' padding, 2x2 strides, RELU, L2 0.001 |
| Convolution2D                 | 4x20x64 - 'same' padding, 2x2 strides, RELU, L2 0.001 |
| Dense                         | 1000 - RELU, L2 0.001                                 |
| Dense                         | 250 - RELU, L2 0.001                                 |
| Dense                         | 50 - RELU, L2 0.001                                 |
| Dense                         | 1                                                   |


Total params: 5,489,371
Trainable params: 5,489,371


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
