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

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/driving_center.png "Driving - center camera"
[image3]: ./examples/driving_right.png "Driving - right camera"
[image4]: ./examples/driving_left.png "Driving - left camera"
[image5]: ./examples/driving_augmented.png "Driving - augmented"
[image6]: ./examples/chart_original_dataset.png "Bar chart - original dataset"
[image7]: ./examples/chart_validation_dataset.png "Bar chart - validation dataset"
[image8]: ./examples/chart_training_dataset.png "Bar chart - training dataset"
[image9]: ./examples/figure_loss.png "Training and validation loss"


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

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 64 and fully connected layers with feature sizes between 50-1000 (model.py lines 264-301)

The model includes RELU layers to introduce nonlinearity, and the data is normalized and cropped in the model using a Keras lambda and cropping layers (model.py lines 271-273).

The model is a somewhat modified version of the [nvidia model](https://arxiv.org/pdf/1604.07316.pdf), tuned to work better for the current problem.

####2. Attempts to reduce overfitting in the model

The model uses L2 regularization to prevent overfitting (e.g. model.py line 266). I also experimented with using Dropout layers, which worked, but using L2 made the loss decrease smoother (not jumping up and down), and the car run smoother on Track 1.

Besides this I also augmented and filtered the data. For details, see the last section.

The model was trained and validated on different data sets (both track 1 and 2) to ensure that the model was not overfitting (code line 321-325).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for both Track 1 and Track 2.

I also modified the speed in drive.py to use the maximum, to see how well the model is able to keep up. It still drives both tracks well, although on Track 2 it gets very close to running off the road a couple times, but manages to (barely) stay on the the road.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 299).

The L2 regularization is set to 0.001, which works quite well in preventing overfitting.

The final model was trained for 15 epochs, which seemed to lead to the best result when testing on both tracks 1 and 2.

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

The final model architecture (model.py lines 264-301) consisted of a convolution neural network with the following layers and layer sizes:


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


Total params: 4,721,371

Here is a visualization of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the udacity provided dataset. I also captured aitional data by driving 4 laps on track 1, but ended up not using it, because the udacity data was better.

Here's an image showing center driving

![alt text][image2]

I didn't record any recovery laps. Instead I was focusing on how to augment the data, so that the model can learn recovery from it. Here are some images used.

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the dataset I did a couple of things:
* Filtered out 2/3 of the data which had 0 angle
* For the remaining 0 angles 1/3 of the time I added either the left or the right image, with adjusted angles.
* For all other data I added both left and right camera images
* In generation time, I flipped 1/2 of the images
* In generation time, I augmented 1/2 of the images, applying random brightness, darkness adjustments, random horizontal transformation and random horizon shifting. This was only necessary for track 2.

#####Angle adjustment

First I arbitrarily chose a number to adjust the angle for the left and right camera images, but after trying a couple numbers in range of [0.15-0.25], it didn't feel, they worked well. So I went on to do some back of the envelope calculations.

To have an estimate of the distance of the cameras, I checked the width of a Toyota Camry online and divided it by 2. I got a distance of *0.915 meters*. Based on the nvidia paper, I chose to calculate an angle needed for the car to return to the center position in 2 seconds. Assuming the car is doing *30mph/50kph*, it can do about *28 meters*. I then added these numbers to an online triangle calculation, and it gave me a result of *1.875 degrees* for the adjustment. Normalizing this the result is 0.075 for the adjustment. I increased this to **0.09** so that the model is recovering a little quicker which might be helpful for sharp turns.

#####Image transformations
Track 1 didn't need additional image augmentation beside the left/right camera images. However track 2 did need more data, because in case of sharp turns or driving down/uphill, the model would try to turn too late or too slowly and run off the road.

Both horizontal transformation and horizon shift was needed. The reason at least partially might be, that the collected data was not high quality enough. I had a problem keeping the car in the middle of the road and in case of sharp turn I often ended up getting very close to the side of the road.

For horizontal transformation instead of just randomly applying left/right transformation for the images, I ended up doing that only for images where the angle was < 0.25. For larger angles I did a random shift so that the angle became even larger. My assumption was, that this might help the model stay closer to the center in case of sharp turns. This helped somewhat, but not nearly as much as I was hoping for.

Another thing I tried to combat the car going off road on sharp corners was, to increase the angle correction. However, this had the effect of the car going off road on the next sharp corner if there was more than one in close proximity, due to overcorrecting when coming out of the first corner.

The horizon shifting helped a lot for the cases when the model started turning too late after going uphill/downhill.

#####The final dataset

Before augmenting the data I first split it to training and validation sets. To do this I simply split the data in two to have 75% as traiing and 25% as validation, without shuffling. The reason for this is, that this is basically a driving session, so our best bet in having a good validation set is to have approx 1 lap on each track (25% is assuming 4 laps in the dataset).

The augmentation was only applied to the training dataset.

Here are the dataset sizes:
Training data: #: (23947,) - augmented
Validation data: #: (4002,) - no augmentation
Original data: #: (16007,) - no augmentation

Basically augmenting the data ~doubled the size of it, even though 2/3 of the 0 angles have been removed.

Here's the original data distribution:
![alt text][image6]

The validation distribution below looks very similar, which is what we want:
![alt text][image7]

And finally the augmented/filtered trainig dataset:
![alt text][image8]

This only contains the left and right images. Flipping and applying additional augmentation was done during generation time, so that the model sees a larger number of examples.

The data was shuffles during generation time every time when the generator restarted (which basically means every epoch in this case), and also the batches were shuffled as well, before yielding.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the learning history below.

![alt text][image9]

Though the model doesn't learn much after ~5th epoch, my observation was, that in order to behave better in sharp turns on track 2, it needed more epochs.

Validation loss is always below training loss because the training set is heavily augmented, while the validation is not. The validation loss is a better indicator of the model accuracy, but based on my experience, the absolute value of the loss doesn't mean much below a certain threshold. The only way to really test whether or not the car goes off road is to test in the simulator.

I used an adam optimizer so that manually training the learning rate wasn't necessary. Nevertheless I tried setting the learning late to 0.0001, but beside making the learning slower, that didn't seem to have any effect.

#####Testing in the simulator
To test the model in the simulator I decided to use the maximum speed after the model was able to pass it at the default speed of 9. In case of track 1 it didn't make much difference, but for track 2 this revealed, that the model is having problems in turns after going uphill/downhill, so in overall this resulted in a better model.

On track 2 the model sometimes gets very close to going off-road, but still manages to steer back at the last minute. At least partially the reason for this might be the dataset used. I'm convinved the model does a better job at driving on track 2 than I did :).

Finally, one problem I was trying to figure out for a long time was, why my model was drving off the road after the bridge. I ruled out the possibility of not having a line on the right side of the road, because there's another place on the track with the exact sideline, and it didn't have a problem there.
Beside the model sometimes was also going to the left side of the road and driving on the sideline.

After a couple days of trying to figure this out I realized, that OpenCV loads the images in BGR, which are fed to the model as is, while the simulator provides images in RGB. After converting the images to RGB in training time the model drove the track perfectly...
