
# Traffic Sign classifier project 


## Data Summary And Exploration



The submission includes:

1) Basic Summary of the data set:
    
    Number of images in the training set: 34799
    Number of images in validation set: 4410
    Number of images in test set: 12630

2) Exploratory visualization
    The number of images per class for each of train, test and validation are plotted.
    Sample images from training set are also displayed.

## Design and Test Model Architecture

The dataset is in RGB and hence preprocessing of the images are done.

### Preprocessing:

First the images and converted to grayscale and then normalized. This reduces the computational burden and normalization prevents loss of information.

### Data Augmentation:

From the plots for the exploratory visualizations, we saw that some classes have far fewer images than others. We use augmentation techniques to help balance the train set.
Steps for augmentation followed are:

1)A threshold of 1000 was chosen( based on an average). 
2)For classes with fewer than 1000 images, we calculate the difference from the threshold.
3)Rotation is used as an augmentation technique.
4)We randomly choose an image from the class, rotate and add it to the dataset. Step 4) is repeated till there are 1000 images in the class.

At the end of this process, all classes have atleast 1000 images.
The counts per class for the training set is plotted and displayed in the submission.


### Model Architecture

#### Details of the characteristics and qualities of the architecture

Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6

Activation: Relu

Pooling. Input = 28x28x6. Output = 14x14x6. padding = VALID

Layer 2: Convolutional. Output = 10x10x16

Activation: Relu

Pooling. Input = 10x10x16. Output = 5x5x16. padding = VALID

Layer 3: Fully Connected. Input = 400. Output = 250

Activation : Relu

Layer 4: Fully Connected. Input = 120. Output = 84

Dropout. Keep prob at 0.85

Layer 5: Fully Connected. Input = 84. Output = 43

#### Describe how the model was trained. 

The model was trained on the augmented data set using the following parameters:

Optimizer: ADAM optimizer

Batch Size: 128

Number of Epochs: 27

Learning rate: 0.00097


#### Describe the approach to finding a solution. Accuracy on the validation set is 0.93 or greater

The model accuracy values:

Validation Accuracy: 98.2%

Test Accuracy: 90.5%

The plot of validation and test accuracy is included in the submission.
The model architecture uses LeNet convolutional neural network provided in LeNet-Lab as a baseline. Dropout layer was added to avoid overfitting. After multiple iterations, the above hyperparameters were chosen for the network.


## Test a Model on New Images

#### Acquiring New Images
A set of 8 German traffic signs from the web is used as new images.
Preprocessing is done on these images. The RGB and the grayscale images are displayed in the submission.
The images from the web have some noise and brightness levels vary too.

#### Performance on New Images
Performance of these images is compared to the test set. 
The accuracy values are:

Test set: 91%

New image set: 62%

The precision and recall charts for the test and new images are also plotted.
In the test set plot, precision values are lower for classes which have fewer number of training images. More augmentation techniques can be used to enhance the data set for these classes and retrain.

In the plot for the new dataset, one of the classes (label number: 28) has 0 value for precision and recall. One of the reasons for this could be that the denominator (true positives + false positives) is 0. This could mean that the image was predicted as a negative. As always, we can better the network by tweaking hyperparameters and adding layers and iteratively improve the accuracy.

#### Model Certainty - Softmax Probabilities

The top 5 softmax probabilities are printed as output in the submission.
The images are also displayed for the top 5 softmax probabilities. We can see that the network is able to classify the images from the web with good confidence, except for the last one.

