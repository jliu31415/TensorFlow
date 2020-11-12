import tensorflow as tf
import numpy as np
#conda install tensorflow
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#shape(train_images) is 60000x28x28; i.e. 60000 images

train_images = train_images/255.0   #normalize values to [0, 1]
test_images = test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=(28, 28, 1)),   #relu returns max(0, x)
    tf.keras.layers.MaxPooling2D(2, 2), #effectively quarters image size
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),   #relu returns max(0, x)
    tf.keras.layers.MaxPooling2D(2, 2), #effectively quarters image size
    tf.keras.layers.Flatten(),  #flattens layer
    tf.keras.layers.Dense(512, activation='relu'),  #512 neurons in hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid') #return final output as a probability vector
    ])

#Adam is a type of stochastic gradient descent
#Cross-entropy is a measure of the difference between two probability distributions for a given random variable or set of events
#Also viewed as number of bits of information needed to communicate between two distributions
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()