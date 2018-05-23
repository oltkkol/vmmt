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

train_x = x_train / 255.0
test_x = x_test / 255.0

# (2) To One-Hot
number_of_classes = len( set(y_train.flatten()) )
train_y = keras.utils.to_categorical(y_train.flatten(), number_of_classes)
test_y = keras.utils.to_categorical(y_test.flatten(), number_of_classes)

# (4) NN
from keras.models import Sequential
from keras.layers import Dense, Dropout

regL2 = regularizers.l2(0.0000002)

model = Sequential()
model.add( Conv2D(128, (3, 3), padding='same', activation="relu", activity_regularizer=regL2, input_shape=train_x.shape[1:]) )
model.add( Dropout(0.61) )
model.add( Conv2D(6, (3, 3), padding='same', activation="relu", activity_regularizer=regL2) )
model.add( Flatten() )

model.add( Dropout(0.5) )
model.add( Dense(64, activation='relu', activity_regularizer=regL2) )
model.add( Dropout(0.31) )

model.add( Dense(number_of_classes, activation='softmax') )

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, 
        train_y, 
        validation_data=(test_x, test_y), 
        epochs=3500, 
        batch_size=32*32,
        callbacks=[plot_losses],
        shuffle=True)


# (5) All evaluation
from sklearn.metrics import classification_report

def print_eval(x, y, title):
    yHat = model.predict_classes(x, batch_size=128, verbose=1)
    report = classification_report(np.argmax(y, axis=1), yHat)
    print("\n\nREPORT ", title, "\n", report)

print_eval(train_x, train_y, "Train")
print_eval(test_x, test_y, "Test")

# not great, but 67.6 %