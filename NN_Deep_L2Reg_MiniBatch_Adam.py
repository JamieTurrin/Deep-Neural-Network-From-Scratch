# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:44:43 2017

@author: James
"""
def NN_Deep_Doc():
    """   
    Implements a fully-connected Deep Neural network.
    
    NOTE: Generally, the use of Adam Optimization and Mini-Batches requires
    smaller learning rate.....
       
    Incorporates L2 Regularization to prevent over training the model, 
    reduces variance. If lambd = 0.0, then L2 Regularization is turned off, 
    otherwise use a value of 0.0 <= lambd <= 1.0
    
    Incorporates mini-batch gradient descent and Adaptive Moment Estimation (Adam)
    Optimization to speed up gradient descent.
    
    NOTE: The use of mini-batches causes this program to be  noticeably slower than
    using just a single batch mode. A larger mini-batch size speeds up program,
    smaller mini-batch size slows program.
    
    Uses sigmoid activation for output layer, and ReLU activation for hidden layers.
    
    Warning: ReLU can lead to large values of W, A, and Z, which in turn cause
    problems with the sigmoid function for the output layer. If Cost is listed
    as NaN or Inf, then turn off ReLU and use Sigmoid for hidden layers. Or, 
    reduce number of iterations, and try a smaller value for lambd.
    
    """
    return


import numpy as np
import matplotlib.pyplot as plt


################## CREATE MINI BATCHES ########################################

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer, typically a power of 2,
                       default is 64
    seed -- random seed for shuffling the data, default is 0
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    import math
    np.random.seed(seed)            # random seed for shuffling data
    m = X.shape[1]                  # number of training examples
    mini_batches = []        # initialize empty list for holding mini batches
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k* mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

######################## INIITIALIZE L LAYERS #################################
def initialize_parameters_adam(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
                  in which layer_dims[0] is number of inputs, nx,
                  and layer_dims[L]=1, is the single-node output layer
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...same dimensions as Wl
                    v["db" + str(l)] = ...same dimensions as bl
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...same dimensions as Wl
                    s["db" + str(l)] = ...same dimensions as bl
                    
    """

    parameters = {}                # dictionary to hold values of W, b, D
    v = {}                         # dictionary to hold values of momentum
    s = {}                         # dictionary to hold value of RMS Prop
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L): # iterate over every layer from 1 to L-1, except last output layer. 
        
        # initialize array W using small random numbers
        #parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        
        # initialize array W using He et al. initialization
        # Speeds up gradient descent, rate of learning
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
        
        # initialize array W using Xavier initialization
        # Speeds up gradient descent, rate of learning
        #parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1./layer_dims[l-1])
        
        # initialize array b to have 0s
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        # initialize momentum (v) and RMS Prop (s) dictionaries (both part of Adam Optimization)
        v["dW" + str(l)] = np.zeros(parameters['W'+str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters['b'+str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters['W'+str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters['b'+str(l)].shape)
        
        # check dims of arrays
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    # end for loop
        
    return parameters, v, s

################### LINEAR PART OF FORWARD PROPAGATION ########################

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation: logistic regression

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b", 
             stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W,A)+b  # logistic regression equation
    
    # if Z is too small (<-709) it causes errors in sigmoid calculation
    # find Z values that are too small and change them to -709
    too_small = np.where(Z<-709)
    Z[too_small] = -709
    
    assert(Z.shape == (W.shape[0], A.shape[1]))  # check dims of arrays
    cache = (A, W, b)  # store A,W,b values in a tuple
    
    return Z, cache

###################### SIGMOID ACTIVATION FUNCTION ############################

def sigmoid(Z):   
    """  
    Compute the sigmoid of Z
    
        Arguments:
        Z -- pre-activation parameter, which is output of logistic equation
    
        Returns:
        A -- sigmoid(Z) 
        activation_cache = Z  useful for back prop.
    """ 
    
    # put Z in a cache 
    activation_cache = Z
    
    # compute sigmoid activation function
    A = 1 / (1 + np.exp(-Z))  # sigmoid of Z
    
    Aone = np.where(A==1.0) # if A=1, causes errors in cost calculation
    A[Aone] = 0.99999  # change A=1 to A=0.99999
    
    Azero = np.where(A==0.0) # if A=0, causes errors in cost
    A[Azero] = 0.000001  # change A=0 to A=0.00001
        
    return A, activation_cache      # return sigmoid of Z and 'sigmoid'

################### ReLU ACTIVATION FUNCTION ##################################

def relu(Z):
    """
    Compute the ReLU function of Z, the linear activation function
    
    Arguments:
    Z -- pre-activation parameter, output of logistic equation
        
    Returns:
    A -- ReLU(Z)
    activation_cache = Z,  cache Z for use in back prop
    """
    # put Z in a cache
    activation_cache = Z
    
    # compute ReLU activation function: ReLU=max(0,Z)
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)  # check shape of A   
    return A, activation_cache  # return A and 'relu' string
        
################# FORWARD ACTIVATION FUNCTIONS ################################

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layers

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    # call linear_forward function:
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid":   
        # call sigmoid function: returns value of A and a string: 'sigmoid'
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu": 
        # call ReLU function: RELU = max(0,Z), returns A and a string: 'relu'
        A, activation_cache = relu(Z)
   
    assert (A.shape == (W.shape[0], A_prev.shape[1])) # check dims of arrays
    
    # add linear_cache or activation_cache strings to cache tuple
    cache = (linear_cache, activation_cache)

    return A, cache

################### FORWARD PROPAGATION OVER L LAYERS #########################

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- input data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep(), contains values for W, b, D
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []  #  create empty list to hold cache dictionaries
    A = X       #  initialize A to be equal to X, that is, A for layer 0 is same as input
    L = len(parameters) // 2     #  // is integer division....number of layers in the neural network
    
    # Compute activation value for each node of each hidden layer
    # assumes each hidden layer uses ReLU activation function
    for l in range(1, L):  # for each hidden layer, from 1 to L-1, do the following:
        A_prev = A 
        
        # call linear_activation_function using values of A from previous node, W and b 
        # values for current node, and ReLU activation function
        A, cache = linear_activation_forward(A_prev, parameters['W'+ str(l)], parameters['b'+ str(l)], 'relu')
                
        # add cache to caches list for each hidden layer of NN
        caches.append(cache)  
    # end for loop
    
    # Compute activation value for last layer of NN
    # using values of A from the final hidden layer, W and b values for
    # the last (current) layer, and sigmoid activation function
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))  # check dims of array
            
    return AL, caches

######################### COMPUTE COST ########################################

def compute_cost(AL, Y, parameters, lambd):
    """
    Implement the cost function

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost + L2 regularization cost
    """
    
    m = Y.shape[1]      # number of training examples

    # Compute cross entropy loss using activation from last hidden layer (AL)
    # and the original Y values.
    cross_entropy = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
    
    # Compute L2 Regularization Cost   
    if lambd > 0.0:
        
        L = len(parameters)//2  # number hidden layers
        summation = 0.0
        
        for l in range(1,L):
            summation = summation + np.sum(np.square(parameters['W'+str(l)]))
        # endfor loop 
            
        L2_cost = summation * lambd / (2*m)
        
    else:
        L2_cost = 0.0
    
    # Cost is cross entropy loss plus L2 loss
    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cross_entropy) + np.squeeze(L2_cost)
      
    assert(cost.shape == ())    # check dims of cost array, should be a scalar, dim (1)
    
    return cost

################# LINEAR PART OF BACK PROPAGATION #############################

def linear_backward(dZ, cache, lambd):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache  # retrieve values of A_prev, W, and b from cache tuple
    m = A_prev.shape[1]   # number of training examples

    # compute gradients:
    dW = 1/m * np.dot(dZ,A_prev.T) + (lambd * W / m)  # lambd*W/m is L2 regularization term
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)  # check dims of arrays
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

##################### RELU FUNCTION DERIVATIVE ################################

def relu_backward(dA, activation_cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z'  stored for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = activation_cache  # retrieve value of Z from cache
    dZ = np.array(dA, copy=True) # just converting dZ to a correct object.
    
    # When Z <= 0, you should set dZ to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)    
    
    return dZ

############## SIGMOID FUNCTION DERIVATIVE ####################################

def sigmoid_backward(dA, activation_cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' stored for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = activation_cache # get value of Z from activation cache.
    
    s, z_cache = sigmoid(Z) # call sigmoid function, z_cache not used here.
    dZ = dA * s * (1-s)  # compute derivative of sigmoid
    
    assert (dZ.shape == Z.shape)
    
    return dZ

#################### ACTIVATION PART OF BACK PROPAGATION ######################

def linear_activation_backward(dA, cache, activation, lambd):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of string values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache  # retrieve linear and activation cache string values from cache tuple 
    
    if activation == "relu":  # for hidden layers
        
        # compute derivative of ReLU activation function
        dZ = relu_backward(dA, activation_cache)
        
    if activation == "sigmoid":   # for output layer
        
        #compute derivative of sigmoid function
        dZ = sigmoid_backward(dA, activation_cache)
    # endif
        
    # compute gradients by calling linear_backward function
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db

################### BACK PROPAGATION OVER L LAYERS ############################

def L_model_backward(AL, Y, caches, parameters, lambd):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    parameters -- dictionary with values of W, b, D
    lambd -- L2 regularization parameter  0.0 <= lambd <= 1.0
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}  # create empty dictionary to  hold gradients dA, dW, db
    L = len(caches) # the number of layers
    # m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # derivative of cost with respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
    
    ################ Initialize back propagation on Lth (final) layer #################
    
    # Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]  # begin with last set of values in caches dictionary, indexed using L-1
    
    # compute gradients on last layer by calling linear_activation_backward function
    # using sigmoid activation function, derivative of A on last layer (dAL)
    # and last set of values in caches dictionary: current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,'sigmoid',lambd)
    
    ################# back propagation on hidden layers ###############################
    
    for l in reversed(range(L-1)):  # proceed backwards thru hidden layers....
        
        # compute lth layer gradients.
        # Inputs gradients of previous hidden layer and cached values: 
            # "grads["dA" + str(l + 2)], caches". 
        # Outputs gradients of current hidden layer: 
            # "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        
        current_cache = caches[l] # retrieve cached values for layer l
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+2)],current_cache,'relu', lambd)
                
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    # end for loop

    return grads

#################### UPDATE PARAMETERS W, b: GRADIENT DESCENT #################

def update_parameters_adam(parameters, grads, v, s, t, learning_rate, layer_dims):
    """
    Update parameters W, b using gradient descent for each hidden layer
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients (dA, dW, db), output of L_model_backward
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    t -- epoch number of mini-batch gradient descent
    learning rate -- scalar
    layer_dims -- list with number of layers in NN and number of neurons in each layer
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2 # number of layers in the neural network
    beta1 = 0.9      # defined some Adam parameters:
    beta2 = 0.999   # standard values for beta1, beta2, epsilon...
    epsilon = 1e-8
    v_corrected = {}   # Initializing first moment estimate, python dictionary
    s_corrected = {}   # Initializing second moment estimate, python dictionary

    # Update parameters W and b of each hidden layer, using Adam Optimization.
    for l in range(L):  # for each hidden layer:
        
        # Moving average of the gradients. 
        v["dW" + str(l+1)] = beta1*v['dW'+str(l+1)] + (1-beta1)*grads['dW'+str(l+1)]
        v["db" + str(l+1)] = beta1*v['db'+str(l+1)] + (1-beta1)*grads['db'+str(l+1)]

        # Compute bias-corrected first moment estimate.
        v_corrected["dW" + str(l+1)] = v['dW'+str(l+1)] / (1-np.power(beta1, t))
        v_corrected["db" + str(l+1)] = v['db'+str(l+1)] / (1-np.power(beta1, t))

        # Moving average of the squared gradients.
        s["dW" + str(l+1)] = beta2*s['dW'+str(l+1)] + (1-beta2)*np.power(grads['dW'+str(l+1)],2)
        s["db" + str(l+1)] = beta2*s['db'+str(l+1)] + (1-beta2)*np.power(grads['db'+str(l+1)],2)

        # Compute bias-corrected second raw moment estimate.
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-np.power(beta2,t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-np.power(beta2,t))
        
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * v_corrected['dW'+str(l+1)]/(np.sqrt(s_corrected['dW'+str(l+1)])+epsilon)
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * v_corrected['db'+str(l+1)]/(np.sqrt(s_corrected['db'+str(l+1)])+epsilon)
    # end for loop
    
    return parameters, v, s

############################ NETWORK MODEL ####################################

def minibatch_adam_model(X, Y, layers_dims, learning_rate, num_iterations, lambd,
                  mini_batch_size, print_cost=False):
    """
    L-LAYER-MODEL FUNCTION:
        
    Implements a fully-connected, L-layer neural network, with ReLU activation functions
    used in hidden layers and sigmoid activation used for a single output
    layer
    
    ARGUMENTS:
        
    X -- training data, numpy array of shape (number of examples, nx)
    
    Y -- true "label" vector (containing 0s and 1s), of shape (1, number of examples)
    
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    
    learning_rate -- learning rate of the gradient descent update rule
    
    num_iterations -- number of iterations/epochs of the optimization loop
    
    lambd -- tuning parameter for L2 regularization, 0.0 <= lambd <= 1.0, 
             lambd=0.0 turns off L2 regularization
             
    mini_batch_size -- size of each mini-batch of data
    
    print_cost -- if True, it prints the cost every 100 steps
    
    RETURNS:
        
    parameters -- parameters (W,b) learned by the model to be used for predictions
    
    """

    costs = []            # create empty list to keep track of cost
    t = 0                 # initializing the counter required for Adam update
    seed = 10             # set an initial seed
    
    # Parameters initialization.
    parameters, v, s = initialize_parameters_adam(layers_dims)

    # Optimization loop
    for i in range(num_iterations):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = L_model_forward(minibatch_X, parameters)

            # Compute cost
            cost = compute_cost(AL, minibatch_Y, parameters, lambd)

            # Backward propagation
            grads = L_model_backward(AL, minibatch_Y, caches, parameters, lambd)

            # Update parameters
            t = t + 1 # Adam counter
            parameters, v, s = update_parameters_adam(parameters, grads, v, s, t, learning_rate, layers_dims)
        # end inner for loop
         
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
    # end outer for loop
            
    # plot the cost as a function of iterations
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (X100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

################## MAKE PREDICTIONS USING TRAINED NETWORK #####################

def predict(parameters, X):
    
    """
    
    PREDICT FUNCTION:
        
    Function to classify test data using parameters obtained from a trained
    Deep_NN.
    
    ARGUMENTS:
        
    parameters -- the parameters dictionary returned by L_layer_model function
    which contains values of W and b
    
    X -- test data, numpy array of shape (number of test samples, nx)
    
    RETURNS:
        
    predictions -- a 1-D array of zeros and ones, corresponding to the two
    categories of classification.
    
    """
    
    # Forward propagation: 
    AL, caches = L_model_forward(X, parameters)
    
    ones = np.where(AL>0.5) # get indices where A2 > 0.5
    
    # initialize array of zeros, same size as output layer, to hold predictions
    predictions = np.zeros((1,X.shape[1])) 
    predictions[ones]=1  #  make predictions=1 where AL > 0.5
    
    return predictions


















