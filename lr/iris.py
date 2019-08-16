import lr

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pprint


# load and process the iris data
# keep only two features (sepal length, sepal width) for easy visualization
def create_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data[:, :-1], data[:, -1]


X_iris, Y_iris = create_iris_data()
Y_iris = np.array([1 if i == 1 else 0 for i in Y_iris]).reshape(-1, 1)
# X_train, X_test, y_train, y_test = train_test_split(X_iris, Y_iris, test_size=0.3) # shuffle and split the data set

# create the model
model = lr.LogisticRegression(max_iter=100, learning_rate=0.1)
model.fit(X_iris, Y_iris)
acc = model.score(X_iris, Y_iris)
print(acc)

print(model.w_)
print(model.b_)