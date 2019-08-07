import autodiff as ad
from util import load_mnist_data
import numpy as np


def load_mnist_for_lr():
    # load and selecte 0, 1
    train_set, valid_set, test_set = load_mnist_data("mnist.pkl.gz")

    # train set
    train_zero_mask = (train_set[1] == 0)
    train_one_mask = (train_set[1] == 1)
    train_mask = train_zero_mask + train_one_mask
    train_X = train_set[0][train_mask]
    train_Y = train_set[1][train_mask].reshape(-1, 1)

    # test set
    test_zero_mask = (test_set[1] == 0)
    test_one_mask = (test_set[1] == 1)
    test_mask = test_zero_mask + test_one_mask
    test_X = test_set[0][test_mask]
    test_Y = test_set[1][test_mask].reshape(-1, 1)

    return train_X, train_Y, test_X, test_Y


def mnist_lr(num_epochs=10, print_loss_val_each_epoch=False):
    print("Build logistic regression model...")

    W = ad.Variable(name="W")
    b = ad.Variable(name="b")
    X = ad.Variable(name="X")
    y = ad.Variable(name="y")

    z = ad.matmul_op(X, W) + b
    y_hat = ad.sigmoid_op(z)

    loss = ad.sigmoidcrossentropy_op(z, y)

    grad_W, grad_b = ad.gradients(loss, [W, b])
    executor = ad.Executor([loss, grad_W, grad_b, y_hat])

    # Read input data
    train_X, train_Y, test_X, test_Y = load_mnist_for_lr()
    print("train set num: %d" % train_X.shape[0])
    print("test set num: %d" % test_X.shape[0])

    # Set up minibatch
    batch_size = 1000
    n_train_batches = train_X.shape[0] // batch_size
    n_test_batches = test_X.shape[0] // batch_size

    print("Start training loop...")

    # Initialize parameters
    W_val = np.zeros((784, 10))
    b_val = np.zeros((10))
    X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, 1), dtype=np.float32)
    test_X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    test_y_val = np.empty(shape=(batch_size, 1), dtype=np.float32)

    lr = 1e-3
    for i in range(num_epochs):
        print("epoch %d" % i)
        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val[:] = train_X[minibatch_start:minibatch_end]
            y_val[:] = train_Y[minibatch_start:minibatch_end]
            loss_val, grad_W_val, grad_b_val, _ = executor.run(
                feed_dict={X: X_val, y: y_val, W: W_val, b: b_val})
            # SGD update
            W_val = W_val - lr * grad_W_val
            b_val = b_val - lr * grad_b_val
        if print_loss_val_each_epoch:
                print(loss_val)

    correct_predictions = []
    for minibatch_index in range(n_test_batches):
        minibatch_start = minibatch_index * batch_size
        minibatch_end = (minibatch_index + 1) * batch_size
        test_X_val[:] = test_X[minibatch_start:minibatch_end]
        test_y_val[:] = test_Y[minibatch_start:minibatch_end]
        _, _, _, test_y_predicted = executor.run(
            feed_dict={
                X: test_X_val,
                y: test_y_val,
                W: W_val,
                b: b_val})
        correct_prediction = (test_y_predicted >= 0.5).astype(np.int) == test_y_val
        correct_predictions.extend(correct_prediction)
    accuracy = np.mean(correct_predictions)
    print("test set accuracy=%f" % accuracy)


if __name__ == '__main__':
    # after 10 epochs, test set acc=1
    mnist_lr(10, True)
