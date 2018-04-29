import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from IPython.display import clear_output

# PLOTTER 
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}): 
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()

plot_losses = PlotLosses()

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# (1) Flatten
newSize = x_train.shape[1] * x_train.shape[2]
train_x = np.reshape(x_train, (len(x_train), newSize))
test_x = np.reshape(x_test, (len(x_test), newSize))

# (2) Scale
train_x = train_x / 255.0
test_x = test_x / 255.0

# (3) To One-Hot
number_of_classes = len( set(y_train) )

train_y = keras.utils.to_categorical(y_train, number_of_classes)
test_y = keras.utils.to_categorical(y_test, number_of_classes)

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
        epochs=5, 
        batch_size=1024,
        callbacks=[plot_losses])

# (5) Try some
def test_example(index):
	probs = model.predict(np.array( [train_x[index]] ))
	yHat = np.argmax( probs )

	plt.matshow( x_train[index], cmap="gray" )
	plt.show()

	print("Target: ", y_train[index], "... Neural network says: ", yHat)

test_example(0)
test_example(1)
test_example(2)