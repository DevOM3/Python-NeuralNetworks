import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist     # loading fashion dataset
# here images are arrays of 28x28

(train_images, train_labels), (test_images, test_labels) = data.load_data()  # loading split data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# shrinking the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.imshow(train_images[7])     # showing images using matplotlib
# plt.show()

# training our neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # flattening the data data=28x28
    keras.layers.Dense(128, activation="relu"),  # giving the dense layer i.e. fully connected layer with rectified linear unit activation function as it is really faster
    keras.layers.Dense(10, activation="softmax")    # softmax activation here it pick value for each neuron to add it to one to et the probability
])
# compiling the module with "adam" optimizer, loss function and "accuracy" metrics
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# fitting and getting our model trained with epochs=10 i.e. 10 times for random data
model.fit(train_images, train_labels, epochs=10)

# evaluating the model to get the accuracy and the loss
# test_loss, test_accuracy = model.evaluate(test_images, test_labels)
# print(test_accuracy)

# predicting the model
prediction = model.predict([test_images])
# showcasing our predictions on the matplotlib visualization
for i in range(10):
    plt.grid = False
    plt.imshow(test_images[i], cmap=plt.cm.binary)  # showing image well
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
