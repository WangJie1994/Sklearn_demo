from sklearn import datasets
from sklearn .cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, random_state=4)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# print(knn.score(X_test, y_test))

from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, iris_X, iris_y, verbose=1, scoring='accuracy', cv=5)
print(scores)
print(scores.mean())


import matplotlib.pyplot as plt
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, iris_X, iris_y, verbose=0, scoring='accuracy', cv=10) # for classification
    loss = -cross_val_score(knn, iris_X, iris_y, verbose=0, scoring='mean_squared_error', cv=10) # for regression
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
