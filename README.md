# homebrew_dnn
## Using nothing but numpy we create a simple Deep Neural Network. Designed for arbitrary inputs, size, depth and activation functions. 

With so many examples already online there is little we can do in the way of improving the hand-coded neural-network. Rather, this is an exersize to show that it can not only be accomplished to handle arbitrarily sized DNNs, but also it can be done with a simple and readible pythonic codebase. With this in mind, I declined to include more advanced features such as mommentum and dropout. 

The DNN here allows us to use tanh and softmax as activation functions, while the cross-entropy loss figure is provided as a cost function. The weight updates are determined by mini-batch stochastic gradient decent. Moreover, because we can caluclate thes gradients with simple matrix manipulations the code runs surprisingly fast without any additional optimization (on an Lenonvo T470S with i7 processor each epoch finishes in 4-5 seconds for our dataset).

To test the Homebrew DNN we used the MNIST handwritted digits dataset (28x28=784 pixels)

The first task was to find an architecture that would preform classification as best as possible. For this I chose two tanh hidden layers (128, 32 nodes) and a softmax output layer (10 nodes).

![](https://raw.githubusercontent.com/gonzodeveloper/homebrew_dnn/master/imgs/Results_01.png)

As you can see the preformance was pretty good. The model then went on to predict the test data with 97.50% accuracy.

Now, to create an architecture that overfits the data. With the same number of epochs we can get the model to overfit by reducing the hidden layers and updating their size. So lets try 1 tanh hidden layer with 5000 nodes, and a softmax output layer (10 nodes).

![](https://raw.githubusercontent.com/gonzodeveloper/homebrew_dnn/master/imgs/Results_02.png)

While we can get high scores on the training data. Once we run this model on the test set, the accuracy falls to 93.81%.



