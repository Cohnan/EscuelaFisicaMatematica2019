############ PHYS 777: MACHINE LEARNING FOR MANY-BODY PHYSICS, TUTORIAL 1 ############
### Code by Lauren Hayward Sierens and Juan Carrasquilla
###
### This code builds a simple data set of spirals with K branches and then implements
### and trains a simple feedforward neural network to classify its branches.
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#Specify font sizes for plots:
plt.rcParams['axes.labelsize']  = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

seed=1234
np.random.seed(seed)
tf.set_random_seed(seed)

plt.ion() # turn on interactive mode (for plotting)

############################################################################
####################### CREATE AND PLOT THE DATA SET #######################
############################################################################

N = 50 # number of points per branch
K = 3  # number of branches

N_train = N*K # total number of points in the training set
x_train = np.zeros((N_train,2)) # matrix containing the 2-dimensional datapoints  // Notice that this isn't flattened: basically a vector of coordinates
y_train = np.zeros(N_train, dtype='uint8') # labels (not in one-hot representation)  // TODO: what is the one-hot representation?

mag_noise = 0.3  # controls how much noise gets added to the data	//TODO: what is this?
dTheta    = 4    # difference in theta in each branch	//TODO: what is this?

### Data generation: ### // Update of x_train and y_train according to a spiral formula
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.01,1,N) # radius
    t = np.linspace(j*(2*np.pi)/K,j*(2*np.pi)/K + dTheta,N) + np.random.randn(N)*mag_noise # theta
    x_train[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
    y_train[ix] = j

### Plot the data set: ###
fig = plt.figure(1, figsize=(5,5))
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=40)#, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('x1')
plt.ylabel('x2')
fig.savefig('spiral_data.pdf')

############################################################################
##################### DEFINE THE NETWORK ARCHITECTURE ######################
############################################################################

##// OUR EQUATIONS ARE: z^l and a^l are 1xn_l+1 ROW VECTORS  for each datapoint, or mxn_l+1 matrix where each row is the corresponding for the ith data point
#//      z^l = a^(l-1) W^l + b^l
#//      a^l = g(z^l)


### Create placeholders for the input data and labels ###
### (we'll input actual values when we ask TensorFlow to run an actual computation later) ###
x = tf.placeholder(tf.float32, [None, 2]) # input data // Each data point, of which now we have None, is made of 2 atributes, HENCE 2 NEURONS IN THE FIRST LAYER: row vector
y = tf.placeholder(tf.int32,[None])       # labels // A vector with no entries

n2 = 4 #// Size of added Layer 2
### Layer 1: ###
W1 = tf.Variable( tf.random_normal([2, n2], mean=0.0, stddev=0.01, dtype=tf.float32) ) #// Changed K by n2
b1 = tf.Variable( tf.zeros([n2]) ) #// Changed K by n2
z1 = tf.matmul(x, W1) + b1 ## // The z for each datapoint is a 1xn_l+1 ROW VECTOR. x = a0
a1 = tf.nn.sigmoid( z1 )


#/// ADITION: Layer 2
W2 = tf.Variable( tf.random_normal([n2, K], mean=0.0, stddev=0.01, dtype=tf.float32) ) 
b2 = tf.Variable( tf.zeros([K]) )
z2 = tf.matmul(a1, W2) + b2 #// Changed x by a1
a2 = tf.nn.sigmoid( z2 )


### Network output: ###
aL = a2  # // Changed output layer to a2

### Cost function: ### // Cross entropy function, TODO: look it up
### (measures how far off our model is from the labels) ###
y_onehot = tf.one_hot(y,depth=K) # labels are converted to one-hot representation
eps=0.0000000001 # to prevent the logs from diverging
cross_entropy = tf.reduce_mean(-tf.reduce_sum( y_onehot * tf.log(aL+eps) +  (1.0-y_onehot )*tf.log(1.0-aL +eps) , reduction_indices=[1]))
cost_func = cross_entropy
#cost_func = tf.reduce_mean(tf.pow(aL - y_onehot, 2))

### Use backpropagation to minimize the cost function using the gradient descent algorithm: ###
learning_rate  = 1.0 # hyperparameter // stepsize in gradient descent?
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func) # // what is this and why is it called that

N_epochs = 20000 # number of times to run gradient descent

##############################################################################
################################## TRAINING ##################################
##############################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer()) 	####### TODO I BELIEVE IT IS HERE WHERE WE TELL

epoch_list    = []
cost_training = []
acc_training  = []

############ Function for plotting: ############
def updatePlot():

    ### Generate coordinates covering the whole plane: ###
    padding = 0.1 # // to make a grid where the points are
    spacing = 0.02 # // to make a grid where the points are
    x1_min, x1_max = x_train[:, 0].min() - padding, x_train[:, 0].max() + padding
    x2_min, x2_max = x_train[:, 1].min() - padding, x_train[:, 1].max() + padding # // y, as in (x, y)
    x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max, spacing),
                         np.arange(x2_min, x2_max, spacing))

    NN_output       = sess.run(aL,feed_dict={x:np.c_[x1_grid.ravel(), x2_grid.ravel()]})
    predicted_class = np.argmax(NN_output, axis=1)

    ### Plot the classifier: ###
    plt.subplot(121)
    plt.contourf(x1_grid, x2_grid, predicted_class.reshape(x1_grid.shape), K, alpha=0.8)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=40)
    plt.xlim(x1_grid.min(), x1_grid.max())
    plt.ylim(x2_grid.min(), x2_grid.max())
    plt.xlabel('x1')
    plt.ylabel('x2')

    ### Plot the cost function during training: ###
    plt.subplot(222)
    plt.plot(epoch_list,cost_training,'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Training cost')

    ### Plot the training accuracy: ###
    plt.subplot(224)
    plt.plot(epoch_list,acc_training,'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Training accuracy')
############ End of plotting function ############

### Train for several epochs: ###
for epoch in range(N_epochs):
    sess.run(train_step, feed_dict={x: x_train,y:y_train}) #run gradient descent
    
    ### Update the plot and print results every 500 epochs: ###
    if epoch % 500 == 0:
        cost = sess.run(cost_func,feed_dict={x:x_train, y:y_train}) ##### // 
        NN_output = sess.run(aL,feed_dict={x:x_train, y:y_train})   ## ///// PREDICTION MADE
        predicted_class = np.argmax(NN_output, axis=1)
        accuracy = np.mean(predicted_class == y_train)
    
        print( "Iteration %d:\n  Training cost %f\n  Training accuracy %f\n" % (epoch, cost, accuracy) )
    
        epoch_list.append(epoch)
        cost_training.append(cost)
        acc_training.append(accuracy)
        
        ### Update the plot of the resulting classifier: ###
        fig = plt.figure(2,figsize=(10,5))
        fig.subplots_adjust(hspace=.3,wspace=.3)
        plt.clf()
        updatePlot()
        plt.pause(0.1)

plt.savefig('spiral_results.pdf') # Save the figure showing the results in the current directory

plt.show()
plt.pause(20)
