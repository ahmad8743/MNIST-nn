## MNIST-nn
As an introduction to deep learning, I thought it would be a good idea to 
build a fundamental understanding of how basic deep learning algorithms 
work under the hood. So, instead of using well-established Python deep 
learning libraries, I took it upon myself to implement gradient descent and
back propagation algorithms on the MNIST digit dataset from scratch using 
vanilla MATLAB (https://yann.lecun.com/exdb/mnist/). After hours of pulling
my hair out while debugging, I finally got it to work with 90% accuracy. 
The entire experience fired off countless lightbulb moments as I finally 
understood the 'how' behind training a computer. I hope this code can serve 
as a good education tool for not only those who are interested in deep 
learning but also those who just want to get a high level overview--I will 
include a lot of comments :).

## Information
Below is a list of all of the files I used and a brief explanation of what 
they do. Reading the description and following along with the code is 
encouraged.
1. CPUnn / GPUnn
* Note: these functions perform the same task, except they run on different
devices in order to develop good device agnostic code that will leverage 
the computing power of a GPU. An nVIDIA CUDA GPU and the MATLAB Parallel 
Computing Toolkit are required to run GPUnn.m, otherwise run CPUnn.m.
* First, we define important activation functions used in our non-linear 
model.
* Next, we process the image data given to us and change it into something
interpretable by our neural network. In this case, we stack all of the
pixels to get a 784x1 column vector representation of our image.
* Then, we run through the training loop. In this training loop, we first 
pass forward to get a prediction from our model. We then use this 
prediction and the actual label to calculate the gradient of our weights
and biases. This allows us to enter the next step which is propagating 
backwards through our network by updating our weights and biases to 
minimize our cost function.
* Lastly, plot results and save.
2. classifer
* Helper method that takes a 28x28 image and returns a prediction. This is 
done by running through a forward pass similar to the one in the training 
loop.
3. evaluation
* Helper method to return our model's confusion matrix and accuracy. Loop 
through east test image and label and use the classifier method to get a 
prediction.