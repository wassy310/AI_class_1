from keras.utils import np_utils
from keras.datasets import mnist

(x_trains, y_trains), (x_tests, y_tests) = mnist.load_data()

x_trains = x_trains.reshape(60000, 28, 28, 1)
x_trains = x_trains.astype('float32')
x_trains /= 255
correct = 10

y_trains = np_utils.to_categorical(y_trains, correct)
x_tests = x_tests.reshape(10000, 28, 28, 1)
x_tests = x_tests.astype('float32')
x_tests /= 255
y_tests = np_utils.to_categorical(y_tests, correct)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(filters = 150,
                 kernel_size = (2, 2),
                 input_shape = (28, 28, 1),
                 padding = 'same',
                 activation = 'relu'
                ))

model.add(
    MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.88))

model.add(Flatten())

model.add(Dense(8192,
                activation='relu'
               ))

model.add(Dense(10,
                activation='sigmoid'
               ))

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
    )

model.summary()

epochs = 20
batchs = 200


history = model.ﬁt(x_trains,
                   y_trains,
                   batch_size=batchs,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(
                       x_tests, y_tests
                   ))

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

plt.ﬁgure(ﬁgsize=(15, 6))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],
         label='training',
         color='black')
plt.plot(history.history['val_loss'],
         label='test',
         color='red')
plt.ylim(0, 0.4)
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(1, 2, 2)
plt.plot(range(1,epochs+1),history.history['accuracy'],
         label='training',
         color='black')
plt.plot(range(1,epochs+1),history.history['val_accuracy'],
         label='test',
         color='red')
plt.ylim(0.5, 1)
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
