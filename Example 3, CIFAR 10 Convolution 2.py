import keras
from keras import *
from keras.layers import *
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import cifar10
from IPython.display import clear_output
from keras.models import Sequential
from scipy.misc import toimage

plt.rcParams["figure.figsize"] = [5.25, 4.25]

# (1) Load CIFAR
###############################################################################

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_x = x_train / 255.0
test_x = x_test / 255.0

# (2) To One-Hot
###############################################################################

number_of_classes = len( set(y_train.flatten()) )
train_y = keras.utils.to_categorical(y_train.flatten(), number_of_classes)
test_y = keras.utils.to_categorical(y_test.flatten(), number_of_classes)

# (3) Define helpful plotters
###############################################################################


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


class PlotImages(keras.callbacks.Callback):
    def __init__(self):
        self.class_examples = {}
        self.class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.number_of_classes = len(self.class_names)

        for classId in range(0, self.number_of_classes):
            self.class_examples[classId] = np.where( y_train == classId )[0]

    def on_epoch_end(self, epoch, logs={}):
        figureIndex = 1

        for classId in range(0, self.number_of_classes):
            exampleIndex = random.choice( self.class_examples[classId] )
            yHats = model.predict( np.array([train_x[exampleIndex]]) )[0]
            yHat = np.argmax(yHats)

            plt.subplot(5, 6, figureIndex, xticks=[], yticks=[])
            plt.imshow(toimage(train_x[exampleIndex], ))
            figureIndex += 1

            plt.subplot(5, 6, figureIndex, xticks=[], yticks=[])
            y_pos = np.arange(len(self.class_names))
            plt.barh(y_pos, yHats.tolist(), align='center', alpha=0.5)
            figureIndex += 1

            if yHat != classId:
                print( self.class_names[classId] + "\t> " + self.class_names[yHat] )

        plt.show()


plot_losses = PlotLosses()
plot_images = PlotImages()

# (4) NN
###############################################################################

# MODEL: 83.2 % @ 400 epoch, parameters: 319k
regL2 = regularizers.l2(0.0000001)

model = Sequential()
model.add( Conv2D(128, (3, 3), padding='same', activation="relu", activity_regularizer=regL2, input_shape=train_x.shape[1:]) )
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add( Conv2D(128, (3, 3), padding='same', activation="relu", activity_regularizer=regL2, input_shape=train_x.shape[1:]) )
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add( Conv2D(128, (3, 3), padding='same', activation="relu", activity_regularizer=regL2, input_shape=train_x.shape[1:]) )
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add( Flatten() )
model.add( Dense(number_of_classes, activation='softmax') )


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

model.fit(train_x, 
        train_y, 
        validation_data=(test_x, test_y), 
        epochs=3500, 
        batch_size=1024,
        callbacks=[plot_losses, plot_images],
        shuffle=True)


# (5) All evaluation
###############################################################################

from sklearn.metrics import classification_report

def print_eval(x, y, title):
    yHat = model.predict_classes(x, batch_size=128, verbose=1)
    report = classification_report(np.argmax(y, axis=1), yHat)
    print("\n\nREPORT ", title, "\n", report)

print_eval(train_x, train_y, "Train")
print_eval(test_x, test_y, "Test")


def plot_confusion(x, y_classes, title):
	matrix = []

	for classId in range(0, number_of_classes):
		indices = np.where(y_classes == classId)[0]
		results = [0]*number_of_classes

		yHats = model.predict_classes(x[indices], verbose=1)
		for i in yHats:
			results[i] += 1

		matrix.append(results)

	plt.imshow(matrix, interpolation='nearest')
	plt.title(title)
	plt.show()


plot_confusion(train_x, y_train, "Train")
plot_confusion(test_x, y_test, "Test")