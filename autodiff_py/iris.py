import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import autodiff as ad


# load and process the iris data
def create_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data[:, :-1], data[:, -1]


X_iris, Y_iris = create_iris_data()
Y_iris = np.array([1 if i == 1 else 0 for i in Y_iris])
# X_train, X_test, y_train, y_test = train_test_split(X_iris, Y_iris, test_size=0.3) # shuffle and split the data set


if __name__ == '__main__':
    W = ad.Variable(name="W")
    b = ad.Variable(name="b")
    X = ad.Variable(name="X")
    y_ = ad.Variable(name="y_")

    z = ad.matmul_op(X, W) + b
    loss = ad.sigmoidcrossentropy_op(z, y_)

    grad_W, grad_b = ad.gradients(loss, [W, b])
    executor = ad.Executor([loss, grad_W, grad_b])

    W_val = np.zeros((2, 1))
    b_val = np.zeros(1)
    X_val = X_iris
    y_val = Y_iris.reshape(100, 1)

    lr = 0.01
    for i in range(100):
        loss_val, grad_w1_val, grad_b_val = executor.run(feed_dict={W: W_val, b: b_val, X: X_val, y_: y_val})
        print(loss_val)
        W_val = W_val - lr * grad_w1_val
        b_val = b_val - lr * np.sum(grad_b_val)

    # we got the same weight with manually working out derivatives and coding them
    print(W_val)
    print(b_val)
