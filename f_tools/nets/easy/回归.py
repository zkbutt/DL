import numpy as np
from keras.models import Sequential
from keras.layers import Dense  ## 全连接层
import matplotlib.pyplot as plt


def 回归结果分析(history):
    # plot loss and accuracy image
    history_dict = history.history  # 拿到完成训练的结果
    train_loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    # figure 1
    plt.figure()
    plt.plot(range(epoch), train_loss, label='train_loss')
    plt.plot(range(epoch), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    # 生成测试数据
    X = np.linspace(-1, 1, 200)
    np.random.shuffle(X)
    Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))

    # 划分训练集和测试集
    X_train, Y_train = X[:160], Y[:160]
    X_test, Y_test = X[160:], Y[160:]

    # start
    model = Sequential()
    model.add(Dense(1, input_dim=1))

    # compile
    model.compile(loss='mse', optimizer='sgd')

    # 训练
    print("\ntraining")
    epoch = 100
    batch_size = 32

    # for step in range(epoch + 1):
    #     cost = model.train_on_batch(X_train, Y_train)
    #     if step % 100 == 0:
    #         print("tarin_cost:", cost)

    ret_train = model.fit(X_train, Y_train,
                          epochs=epoch, batch_size=batch_size,
                          validation_data=(X_test, Y_test),
                          # callbacks=[stopping1],
                          # callbacks=[stopping2],
                          verbose=0,  # 1-显示每批 2-显示每epochs
                          )
    回归结果分析(ret_train)

    # 测试
    print("\nTest")
    cost = model.evaluate(X_test, Y_test, batch_size=40)
    W, b = model.layers[0].get_weights()
    print("Weights", W, "biaxes", b)

    # 预测结果
    Y = model.predict(X_test)
    plt.scatter(X_test, Y_test)
    plt.plot(X_test, Y)
    plt.show()
