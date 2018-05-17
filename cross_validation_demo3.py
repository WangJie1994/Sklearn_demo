from sklearn.learning_curve import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target
param_rage = np.logspace(-6, -2.3, 5)

train_loss, test_loss = validation_curve(SVC(), X, y, param_name='gamma', param_range=param_rage, cv=10, scoring='accuracy')

train_loss_mean = np.mean(train_loss, axis=1)
test_loss_mean = np.mean(test_loss, axis=1)

plt.figure()
plt.plot(param_rage, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(param_rage, test_loss_mean, 'o-', color='g', label='Cross validation')
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()