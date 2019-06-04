import pandas as pd
import numpy as np

from sklearn import linear_model # import thư viện sklearn.linear_model.LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


data_1 = pd.read_csv("data_1.csv")
print(len(data_1))
data_1 = data_1[(data_1['0.1'] < 300) & (data_1['0.1'] > 10)]
print(len(data_1))


X = data_1.loc[:, ['0']].as_matrix() # Sử dụng "cot dau tien" làm biến giải thích
Y = data_1['0.1'].as_matrix() # Sử dụng "cot thu hai" làm biến mục đích

model = linear_model.LinearRegression()

# Tạo model suy đoán
result = model.fit(X, Y)

# Hệ số hồi quy
print("Hệ số hồi quy: a =", result.coef_)

# Sai số
print("Sai số: b = ", result.intercept_)

print("-----------------------------------------")

print("====> Y =", result.coef_,"X +", result.intercept_)
print("-----------------------------------------")

# Score
print("Score:", result.score(X, Y))
print("-----------------------------------------")

# Biểu diễn sự phân bố tập dữ liệu input
# c: color
# Đường thẳng hồi quy

plt.figure(1)
plt.scatter(X, Y, c='r')
plt.plot(X, result.predict(X))
#plt.show()


print("-----------------------------------------")

# Split the data into training/testing sets
X_train = X[:-100]
X_test = X[-100:]

# Split the targets into training/testing sets
y_train = Y[:-100]
y_test = Y[-100:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)
regr2 = linear_model.LinearRegression()
regr2.fit(X_test,y_test)

# Hệ số hồi quy
print("Hệ số hồi quy: a2 =", regr2.coef_)

# Sai số
print("Sai số: b2 = ", regr2.intercept_)

print("-----------------------------------------")

print("====> Y_test =", regr2.coef_,"X_train +", regr2.intercept_)
print("-----------------------------------------")

# Score
print("Score_test:", regr2.score(X, Y))
print("-----------------------------------------")


# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: a: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs

plt.figure(2)
plt.scatter(X_test, y_test,  color='black')
#plt.plot(X_test, y_test, color='red', linewidth=3)
plt.plot(X_test, y_pred, color='blue', linewidth=3)


#plt.xticks(())
#plt.yticks(())

plt.show()