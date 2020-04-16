# Final Project

### The Model:

The classifier we chose for this project is a Convolutional Nerual Net (CNN). These tend to perform well in image-recognition tasks and we were able to achieve very close to 90% testing accuracy without doing a whole lot of pre-processing. Our CNN has 3 convolutional layers and uses a kernel size of 3 in each layer, followed by two linear layers, then the output layer which uses the softmax activation function. We chose Cross-Entropy loss to be the loss function. 

After transforming the provdied data to be grayscale, encoding outputs to be integers 1-9, and partitioning into training and validation sets, we trained our model for 1000 epochs using a batch size of 128 and an initial learning rate of 0.001. The pre-trained model we load in the TestFunction is called CNN_23 and was trained in this fashion.

### The Pipeline:

In our pipeline (the code file is called *ASL_Classifier.ipynb*), we first import the required libraries, then define the CNN class for the CNN described above, and then we define the following functions:

##### PreProcess(data, labels):

This function pre-processes the input images and labels. The first parameter is an Mx100x100x3 matrix of images and the corresponding labels for the images. In the function we transform the images to be Mx100x100 grayscale images. We then transform the labels by encoding the outputs to be integers. The function returns the transformed images and labels as the tensors X_tn and y_tn respectively.

##### TrainModel(CNN, BatchSize, Epochs, Init_LR, X, y):

The inputs to this function are the CNN object to be trained, the batch size to be used (the number of data samples to pass before adjusting weights), the number of epochs (number of times we pass the full dataset), the initial learning rate (which we generally keep at 0.001), and then the transformed images and labels. 

To use this function, all of the input parameters must be specified. So, you would first need to generate an untrained model by calling the CNN class, then calling TrainModel with the parameters you want. We found through testing that batch sizes in powers of 2 between 64-256 worked well and that the number of epochs should be greater than the batch size. For the X and y inputs, pass the training set to PreProcess and then use the outputs from that function as inputs to this function.

##### EvaluateModel(model, X, y):

This function evaluates a passed-in model on a passed-in dataset/labels and returns the accuracy as well as the predicted outputs as numbers. The model parameter can either be a pre-trained model which gets loaded in elsewhere, or it can be a model trained by the previous TrainModel function. As in the previous function, X and y should also be pre-processed images and labels.

##### PostProcess(labels, nums):

This function will take the predicted-labels-as-numbers output of the EvaluateModel function as well as the original labels and does the following: it first applies to the original labels the same transformation used in the PreProcess function so that the LabelEncoder object can then do inverse_transform on the predicted-labels-as-numbers. This allows to transform the predicted labels from numbers to their corresponding letters. We then return these letters.

#### TrainFunction(X, y):

This is one of the two main functions. It will take in the training data as an Mx100x100x3 matrix of images with their corresponding labels. It then does pre-processing on that dataset using the PreProcess function, returning X_train and y_train as tensors. We then initialize an untrained model and then proceed to call the TrainModel function with the same parameters that we used in generating our final model. We then rename the newly-trained model and return it from this function.


#### TestFunction(X, y):

This is the other of the two main functions. It takes in the test data as an Mx100x100x3 matrix of images with their corresponding labels. It then does pre-processing on that dataset using the PreProcess function, returning X_test and y_test as tensors. Next we load in our pretrained model. If we want to use a different model, we can load one that had been previously saved via torch.save() or we can use a newly-trained model from the output of the TrainFunction. We then get the testing accuracy and predicted-labels-as-numbers by calling the EvaluateModel function. Finally, we use the PostProcess function to transform the predicted-labels-as-numbers to letters, and then return the aforementioned testing accuracy and predicted-labels-as-letters.