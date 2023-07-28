import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from lin_regression import LinearRegression

X, y = datasets.make_regression(
    n_samples=100,  n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# show the plot of all data
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color="b",  marker="o", s=30)
plt.show()


reg = LinearRegression(learning_rate=0.01)
# train the program on training data
reg.fit(X_train, y_train)
# create predictions on testing data
predictions = reg.predict(X_test)


# find average error in predictions on testing data
def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)


mse = mse(y_test, predictions)
print(mse)

# show the plot of training data in green, the plot of testing data in teal, and the prediction line in black
y_hat_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test,  y_test, color=cmap(0.5), s=10)
plt.plot(X, y_hat_line, color='black', linewidth=2, label='Prediction')
plt.show()
