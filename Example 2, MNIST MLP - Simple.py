import keras
import numpy as np
from keras.datasets import mnist

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# (1) Scale and Flatten
train_x = []
test_x = []

def scale_and_flatten(example):
	return (example / 255.0).flatten()

for example in x_train:
	train_x.append( scale_and_flatten(example) )

for example in x_test:
	test_x.append( scale_and_flatten(example) )

# (2) Target labels to one-hot
# Eg. 0 to [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

train_y = []
test_y = []
number_of_classes = 10

def target_to_onehot(number):
	template = [0] * number_of_classes
	template[number] = 1
	return template

for target in y_train:
	train_y.append( target_to_onehot(target) )

for target in y_test:
	test_y.append( target_to_onehot(target) )

# (3) To np.array

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = np.array(test_x)
test_y = np.array(test_y)

# (4) NN
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(128, input_dim=len(train_x[0]), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(60, activation='relu'))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, 
        train_y, 
        validation_data=(test_x, test_y), 
        epochs=1000, 
        batch_size=32)