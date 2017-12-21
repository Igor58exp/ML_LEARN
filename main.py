import numpy as nmp
from sklearn import tree
from sklearn.datasets import load_iris

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
classifire = tree.DecisionTreeClassifier()
classifire = classifire.fit(features, labels)
print(classifire.predict([[150, 0]]))

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
for i in range(len(iris.target)):
    print(i, " ", iris.target[i], " ", iris.data[i])

'''
rnd = nmp.random.RandomState(seed = 123)
x = rnd.uniform(low = 0.0, high = 1.0, size = (3, 5))
print(x)
print(x[:, 2])
y = nmp.linspace(0, 12, 5)
print(y)
'''