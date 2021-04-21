# Number$ Classifier 
### _A User Friendly CNN-based Classifier for (poorly) Witten Digits (0-9)_

Number$ Classifier is, as the title suggests, a convolutional neural network that classifies handwritten digits. The 
workflow itself follows a pretty simple structure, but it's worth taking a moment to get familiar with it before trying
to put it to use. 

Note - this file is Markdown. To best view this README, view in a Markdown viewer (such as GitHub)

## Table of Contents
1. [Introduction](#introduction)
2. [Before Getting Started](#section2)  
	2.1. [Libraries](#subsection21)  
	2.2. [Directory Set-Up](#subsection22)
3. [IDE Use](#section3)
4. [Step 1: Training the Algorithm](#section4)  
	4.1. [Input Training Data and Associated Labels](#subsection41)  
	4.2. [Preprocessing Workflow](#subsection42)  
	4.3. [CNN Workflow](#subsection43)  
	4.4. [Output Model](#subsection44)  
5. [Step 2: Running Testing Data and Evaluating Algorithm](#section5)  
	5.1. [Input Testing Data and Associated Labels](#subsection51)  
	5.2. [Outputs: Accuracy Values](#subsection52)  
	5.3. [Outputs: Confusion Matrix](#subsection53)  
	5.4. [Outputs: Predicted Labels](#subsection54)  
6. [Hyperparameters](#section6)  
	6.1. [Preprocessing Parameters](#subsection61)  
	6.2. [Convolutional Neural Network (CNN) Parameters](#subsection62) 
7. [Common Errors (and their solutions!)](#section7)
 

## 1. Introduction <a name="introduction"></a>
___
This number classifier is designed to train on an input dataset made up of handwritten digits (0-9) and their associated labels 
(the training data), and then accurately classify additional handwritten digits (the testing data). Ideally, the dataset in question 
contains several thousand of these handwritten digits, with an equal distribution of each individual digit. This is similar in nature 
to the [MNIST](https://keras.io/api/datasets/mnist/) data and associated classifiers, with a few distinct differences. This program has been developed to
classify digits that are not as neatly written with much more background noise and distortions than the clear examples that can be found 
in MNIST. Additionally, this program's inputs require a higher resolution image input (300x300), which is closer to what one can expect from 
modern photographs of handwriting. However, much of the workflow is similar to MNIST solutions and other number classifiers, as this is a
common exercise in machine learning. 

The programming language used in this program is Python, and therefore the scripts (TRAIN.py and TEST.py) require a Python IDE (Integrated 
Development Environment) to run. A list of such IDEs is available [here](https://www.programiz.com/python-programming/ide).

Finally, this program can be computationally expensive. Depending on your system, it may be appropriate (or at least, recommended) to close all non-essential
programs while running through this classification. Training and testing combined can take as long as 20 minutes (if not longer) depending on the dataset.



## 2. Before Getting Started <a name="section2"></a>
___
_"A little action often spurs a lot of momentum"_ [-Noah Scalin](https://www.goodreads.com/author/show/1360727.Noah_Scalin)
### 2.1. Libraries <a name="subsection21"></a>
Number$ Classifier requires a series of Python libraries to run. Take a look through the following list to make sure you 
have the appropriate libraries loaded in to your system before getting started:

```py
cv2                             conda install -c conda-forge opencv
numpy                           conda install -c anaconda numpy
numpy.matlib, matplotlib.pyplot conda install -c conda-forge matplotlib
seaborn                         conda install -c anaconda seaborn
itertools                       conda install -c anaconda more-itertools
keras                           conda install -c conda-forge keras
time                            conda install -c conda-forge time
```
### 2.2. Directory Set-Up <a name="subsection22"></a>
In addition to the listed libraries, Number$ Classifier also requires certain contents in its directory to run properly. If
You do not know how to set your directory, there are a lot of [great resources](https://linuxize.com/post/python-get-change-current-working-directory/) around the internet that can help you get more familiar 
with your chosen IDE. Your directory must contain:
```sh
Train.py - This will train the CNN on the dataset (will generate 'Number$_CNN_Model.h5' as an output)
Test.py - This will test the trained model ('Number$_CNN_Model.h5')
data_train.npy - Training data (recommend 70-80% of your dataset)
labels_train.npy - Corresponding training data labels
data_test.npy - Testing data (remaining 30-20% of your dataset)
labels_test.npy - Corresponding testing data labels
```
If your images are not in a numpy array format, that's ok! There's a script (provided by Dr Catia Silva) that can convert 
your image files in to a usable format. First give a number from 1 to 4 to each team member (this is the ID). Then, for
example, when team member with ID 4 is recording hers/his 5th handwriting of digit 8, the file name should read "4-5-8.jpg".
Then, run the folder containing the numbers through the following script

```py
labels = np.array([])

i=0
for file in os.listdir(mydir):
    if file.endswith(".jpg"): # Will only read .jpg files (you can change this to other formats)
        filename = mydir+'/'+file
        # Loads image, converts to grayscale and resizes it to a 300x300 image
        y = np.array(Image.open(filename).convert('RGB').convert('L').resize((300,300)))
        
        # Resizes 300x300 image to 90,000x1 array
        col_y = y.ravel()[:,np.newaxis]
        
        # Saves
        if i==0:
            data = col_y
        else:
            data = np.hstack((data, col_y))
        
        # Creates labels from filename
        labels = np.hstack((labels, int(file[-5]))) # this assumes the file extension has 3 letters (such as jpg)
        
        i+=1

print('-------------------------------------------------------')
print('----------------------DONE-----------------------------')
print('-------------------------------------------------------')
print('There are ', data.shape[1],' images')
print('There are ', labels.shape[0],' labels')

# Saves the files to your current directory
np.save('data', data)
np.save('labels', labels)
```

### 3. IDE Use <a name="section3"></a>
___
This program was written for use in a Python IDE, and will work in most publicly available programs. However, it is
strongly recommended that Spyder (from Anaconda) or an equivalent is used.

### 4. Step 1: Training the Algorithm <a name="section4"></a>
___
Upon download, the algorithm is nothing more than a couple hundred lines of code that, by itself, can't classify digits. This
is where the training comes in. Using a [majority](#subsection22) of the collected data as an input, this process will train the model on
how to assign labels to the given images. 
```py
def train_model():
    X_training, y_training = load_trainingData()
    X_training, y_training = scale_trainingData(X_training, y_training)
    datagen = ImageDataGenerator(rotation_range=15, zoom_range = 0.2, width_shift_range=0.2, height_shift_range=0.2)  
    datagen.fit(X_training)
    model = cNN_model()
    model.fit(datagen.flow(X_training, y_training,
          batch_size=batch_size),
          epochs=epochs,
          verbose=0,
          steps_per_epoch=X_training.shape[0] // batch_size)
    model.save('Number$_CNN_Model.h5') 
```

#### 4.1. Input Training Data and Associated Labels <a name="subsection41"></a>
The first time you, the user, will input text in to the code will be when you assign your training data to the variable 'data_train'
```py
# Input training data and labels here

data_train = np.load('TRAIN_data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('TRAIN_labels_train.npy')
print(labels.shape)
```
The ```'TRAIN_data_train.npy'``` represents the file name of your training data, as named in your [directory](#subsection22). This should contain 70-80% of
all the pictures of digits you have collected in 300x300px resolution (If this is not the case, click [here](#subsection22) to see how to set up your data). Go ahead 
and enter the name of your training data _exactly as it appears in your directory_, with the .npy file extension.

This model has been developed to process training data that is in the shape (number of dimensions, number of samples). If you are unsure of the shape of your data,
use the ```_.shape``` command to assess what you're working with. If your data has its rows/columns flipped, simply change the 
```py
data_train = np.load('TRAIN_data_train.npy')
``` 
to
```py
data_train.T = np.load('TRAIN_data_train.npy')
```
Once your training data is in the proper orientation, you are ready to run the TRAIN.py program!


#### 4.2. Preprocessing Workflow <a name="subsection42"></a>  
After importing the data, TRAIN.py preprocesses it to better enable the classification process. This consists of three steps (morphological transformations),
'Thinning', 'Resizing', and 'Outlining' (you can read more about the specifics of each of these functions [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)).

These three transformations are sequential, meaning that they have been programmed and their [hyperparameters](#subsection61) optimized with their specific order in mind. First, the Thinning function (cv2's erode tool) removes background noise from each image, and is immediately followed by each individual image being resized from 300x300px to 20x20px. Finally, each resized image is run through the Outline function, which, by applying a morphological gradient, both limits the remaining background noise and highlights the lines that make up each digit. 

The output of these three functions is a new dataset called 'data_preprocessed.npy', which will appear in your directory once the preprocessing has successfully completed. 

#### 4.3. CNN Workflow <a name="subsection43"></a>  
Following the preprocessing, the training data undergoes a random data augmentation and is fed through a 7 layer convolutional neural network. The data augmentation (which is done through Keras' [ImageDataProcessing](https://keras.io/api/preprocessing/image/)) randomly adjusts each image in the dataset in rotation, zoom, and vertical/horizontal shift 
(as shown below). These shifts are adjustable, and can be manipulated at minimal risk to the algorithm's functionality (though any changes may impact the accuracy of 
the model). 
Data augmentation:
```py
datagen = ImageDataGenerator(
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2)  # randomly shift images vertically (fraction of total height)
```

The CNN architecture is as follows:

In -> [[Conv2D->relu]*2 -> MaxPooling2D]*2 -> [[Conv2D->relu]*2 -> [Conv2D->sigmoid] -> MaxPooling2D]*3 Flatten -> [Dense -> Dropout]*2 
-> Softmax -> [Optimize->Adagrad] -> Compile

The CNN does have adjustable hyperparameters (which have been optimized for this model). These (and their descriptions) can be located [here](#subsection62).  

The kernel sizes for the MaxPooling have been set to (2,2) with strides of 1 or 2 throughout, and the filter sizes for the Conv2D functions vary from 20 to 128, with kernels of (5,5), (3,3), or (1,1).

#### 4.4. Output Model <a name="subsection44"></a>  
After the model has been successfully trained, it will be saved in to the user's directory as 'Number$_CNN_Model.h5'. This program will also, as a console output,
report the time (in seconds) that it took to train your data.

### 5. Step 2: Running Testing Data and Evaluating Algorithm <a name="section5"></a>
___
Once the model has been trained at you have 'Number$_CNN_Model.h5' in your directory (and _only_ once you have 'Number$_CNN_Model.h5' in your directory), you're ready
to test the model! This process uses the trained model generated in TRAIN.py and assesses its ability to classify the [remainder of your dataset](#subsection22).

```py
def test_model():
    X_testing, y_testing = load_testingData()
    X_testing, y_testing = scale_testingData(X_testing, y_testing)
    model = load_model('Number$_CNN_Model.h5')
    score = model.evaluate(X_testing, y_testing, verbose=0)
    print('Test loss:', '%.3f' % (score[0]*100),'%')
    print('Test accuracy:', '%.3f' % (score[1]*100),'%')
    
    Y_pred = model.predict(X_testing)
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    Y_true = np.argmax(y_testing,axis = 1) 
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    plot_confusion_matrix(confusion_mtx, classes = range(10)) 
```

#### 5.1. Input Testing Data and Associated Labels <a name="subsection51"></a>  
```py
data_test = np.load('TEST_data_test.npy')
print('raw data shape', data_test.shape)
labels = np.load('TEST_labels_test.npy')
print(labels.shape)
```

Like TRAIN.py, the ```'TEST_data_test.npy'``` represents the file name of your testing data, as named in your [directory](#subsection22). This should contain 
30-20% of all the pictures of digits you have collected in 300x300px resolution. Enter the name of your testing data _exactly as it appears in your directory_, 
with the .npy file extension.

Also (just like TRAIN.py), this model has been developed to process testing data that is in the shape (number of dimensions, number of samples). If you had to flip
the data in the TRAIN.py function, you'll have to do the exact same thing here
```py
data_test = np.load('TEST_data_test.npy')
``` 
to
```py
data_test.T = np.load('TEST_data_test.npy')
```

Unlike TRAIN.py, however, this program does NOT train the algorithm. Instead, it imports the model generated [here](#subsection44) and evaluates it using the images
and associated labels in the test dataset. 

#### 5.2. Outputs: Accuracy Values <a name="subsection52"></a>  
After running TEST.py, there are a series of console outputs generated. First, the program will print the 'Test Loss', which indicates how well or poorly the model's
prediction was on the test set of data (it is a sum of errors in the test set of data, the lower the better) and the 'Test Accuracy', which represents the ratio of
correctly versus incorrectly classified digits (1.00 being perfect accurate, 0.00 being perfectly inaccurate).

#### 5.3. Outputs: Confusion Matrix <a name="subsection53"></a>  
TEST.py generates a confusion matrix to accompany the accuracy values provided. This figure shows the true labels of each sample versus their labels, as predicted by
the model. The diagonal of this matrix represents the correct label assignment, while any values other than '0' in non-daigonal cells identifies not only which digits
were mis-labeled, but what they were incorrectly labeled as. 

#### 5.4. Outputs: Predicted Labels <a name="subsection54"></a>  
The final output of TEST.py is the list of predicted labels for the test set of data. 


### 6. Hyperparameters <a name="section6"></a>
___
#### 6.1 Preprocessing Parameters <a name="subsection61"></a>
There are three preprocessing hyperparameters that have been tuned for this model, "thin_kernel", "dimensions", and "outline_kernel". Each one 
of these parameters corresponds to one of the three components to the data preprocessing workflow. 
 - thin_kernel: (optimized at 60) dictates the size of the kernel used by the cv2.erode function. A kernel size of 60x60 is used here.
 - dimensions: (optimized at 20) dictates the new number of pixels that the input data will be resized to. A size of 20x20 pixels is used here.
 - outline_kernel: optimized at 4) similar to the thin_kernel, dictates the size of the kernel used for the application of a morphological 
gradient. A kernel size of 4x4 is used here.


NOTE: Changing the 'dimensions' parameter will impact the CNN's parameters as well. Avoid doing so at the risk of compromising the convolutional layers!
```py
# Preprocessing Dimensions

thin_kernel = 60
dimensions = 20
outline_kernel = 4
```

for more information on these hyperparamters and their use in the preprocessing workflow, visit the [OpenCV Morphological Transformations page](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)

#### 6.2 Convolutional Neural Network (CNN) Parameters <a name="subsection62"></a>
After the data has been preprocessed, it is fed in to the CNN to either train or test the algorithm. This neural network has its own series of 
hyperparameters that have been carefully optimized to yield the strongest results for the digit classification. 
- num_classes: The final number of classes the CNN must sort the data it to. As there are 10 digits (0-9), this value will stay at 10
- img_rows, img_cols: set to the resize dimensions (20x20). This drives the CNN's rearranging of the data pixels to create a square that
can support 2D convolutional layers later in the program.
- batch_size: The batch size is a number of samples processed before the model is updated
- epochs: The number of epochs is the number of complete passes through the training dataset. This is the maximum number of epochs that 
will be run, though the program will stop early at convergence if necessary
- cNN_NN_layer1: The number of neurons in the first hidden layer of the neural network
- cNN_NN_layer2: The number of neurons in the second hidden layer of the neural network

```py
# c-NN Hyperparameters
num_classes = 10
img_rows, img_cols = dimensions, dimensions
batch_size = 200
epochs = 400
cNN_NN_layer1 = 1500
cNN_NN_layer2 = 1000
```  

### 7. Fixes to Common Errors <a name="subsection7"></a>
shape of input data
number of preprocessing methods used
Keras issues due to dimensionality changes











#### END OF README
