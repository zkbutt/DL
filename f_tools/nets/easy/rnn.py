import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense  ## 全连接层
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
index_start = 0
OUTPUT_SIZE = 10
CELL_SIZE = 75
LR = 1e-3

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28) / 255
X_test = X_test.reshape(-1, 28, 28) / 255

Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

model = Sequential()

# conv1
model.add(
    SimpleRNN(
        batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
        output_dim=CELL_SIZE,
    )
)
model.add(Dense(OUTPUT_SIZE))
model.add(Activation("softmax"))
adam = Adam(LR)
## compile
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

## tarin
for i in range(500):
    X_batch = X_train[index_start:index_start + BATCH_SIZE, :, :]
    Y_batch = Y_train[index_start:index_start + BATCH_SIZE, :]
    index_start += BATCH_SIZE
    cost = model.train_on_batch(X_batch, Y_batch)
    if index_start >= X_train.shape[0]:
        index_start = 0
    if i % 100 == 0:
        ## acc
        cost, accuracy = model.evaluate(X_test, Y_test, batch_size=50)
        ## W,b = model.layers[0].get_weights()
        print("accuracy:", accuracy)
