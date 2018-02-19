import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
session = tf.Session()
print(session.run([node1, node2]))
session.close()


'''
import inspect
import numpy as nmp
import random
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from sklearn.externals.six import StringIO
import pydot_ng as pydot
import matplotlib.pylab as plt

def lineNumber():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno
'''
'''
def euc(a, b):
    return distance.euc(a, b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
'''
'''
######   Pipeline  --------------------------------------------------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)
my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)
print(accuracy_score(y_test, predictions))


'''
'''
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
classifire = tree.DecisionTreeClassifier()
classifire = classifire.fit(features, labels)
print(classifire.predict([[150, 0]]))
'''

'''
iris = load_iris()
print(lineNumber(), iris.feature_names)
print(lineNumber(), iris.target_names)
print(lineNumber(), iris.data[0])
#for i in range(len(iris.target)):
    #print(i, " ", iris.target[i], " ", iris.data[i])

testIndex = [0, 50, 100]
#training data
trainingTarget = nmp.delete(iris.target, testIndex)
trainingData = nmp.delete(iris.data, testIndex, axis = 0)
#testing data
testTarget = iris.target[testIndex]
testData = iris.data[testIndex]
classifier = tree.DecisionTreeClassifier()
classifier.fit(trainingData, trainingTarget)
print(testTarget)
print(classifier.predict(testData))

dotData = StringIO()
tree.export_graphviz(classifier,
                     out_file = dotData,
                     feature_names = iris.feature_names,
                     class_names = iris.target_names,
                     filled = True,
                     rounded = True,
                     impurity = False)
graph = pydot.graph_from_dot_data(dotData.getvalue())
print(testData[0], testTarget[0])
'''


'''
greyhounds = 500
labs = 500
grey_height = 28 + 4 * nmp.random.randn(greyhounds)
lab_height = 24 + 4 * nmp.random.randn(labs)
plt.hist([grey_height, lab_height], stacked = True, color = ['r', 'b'])
plt.show()
'''

'''
rnd = nmp.random.RandomState(seed = 123)
x = rnd.uniform(low = 0.0, high = 1.0, size = (3, 5))
#print(x)
#print(x[:, 2])
y = nmp.linspace(0, 12, 5)
'''
