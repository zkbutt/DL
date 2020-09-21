import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten  ## 全连接层
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

model = Sequential()

# conv1
model.add(Conv2D(filters=32, kernel_size=5, padding='same', input_shape=(28, 28, 1)))
model.add(Activation("relu"))
# pool1
model.add(MaxPooling2D())

# conv2
model.add(Conv2D(filters=64, kernel_size=5, padding='same', input_shape=(28, 28, 1)))
model.add(Activation("relu"))
# pool2
model.add(MaxPooling2D())

# 全连接层
model.add(Flatten())  # 展平层
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))

adam = Adam(lr=1e-4)

## compile
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

# tarin
print("\ntraining")
history = model.fit(
    X_train, Y_train, epochs=2, batch_size=32,
    validation_data=(X_test, Y_test),
)

print("\nTest")
## acc
cost, accuracy = model.evaluate(X_test, Y_test)
## W,b = model.layers[0].get_weights()
print("accuracy:", accuracy)
