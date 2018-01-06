# Deep-Neural-Network-From-Scratch
Python program which implements a deep neural network from scratch

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
