import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from IPython.display import clear_output
from keras import *
from keras.layers import *

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

# Load CIFAR

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# (1) Flatten & scale
def flatten_and_scale(example):
    return example.flatten() / 255.0

train_x = []
test_x = []

for example in x_train:
    train_x.append( flatten_and_scale(example) )

for example in x_test:
    test_x.append( flatten_and_scale(example) )

train_x = np.array(train_x)
test_x = np.array(test_x)

# (2) To One-Hot
number_of_classes = len( set(y_train.flatten()) )
train_y = keras.utils.to_categorical(y_train.flatten(), number_of_classes)
test_y = keras.utils.to_categorical(y_test.flatten(), number_of_classes)

# (4) NN
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(3072, input_dim=len(train_x[0]), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, 
        train_y, 
        validation_data=(test_x, test_y), 
        epochs=3500, 
        batch_size=32000,
        callbacks=[plot_losses])

# (5) All evaluation
from sklearn.metrics import classification_report

def print_eval(x, y, title):
    yHat = model.predict_classes(x, batch_size=128, verbose=1)
    report = classification_report(np.argmax(y, axis=1), yHat)
    print("\n\nREPORT ", title, "\n", report)

print_eval(train_x, train_y, "Train")
print_eval(test_x, test_y, "Test")

# not good at all, lets add convolution