import loaddata_lab5_mai_andrew as loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

training_percentage = 0.3

data = loader.load("data_lab5.csv")
data_expected = pd.read_csv(
    "data_lab5_expanded.csv")

print(len(data), len(data_expected))

training_index = int(len(data) * training_percentage)

training_set = data_expected.loc[:training_index]
train_features = training_set.drop('TARGET_D', axis=1, inplace=False)
training_label = training_set.loc[:, 'TARGET_D']

test_set = data_expected.loc[training_index + 1:]
test_features = test_set.drop('TARGET_D', axis=1, inplace=False)
test_label = test_set.loc[:, 'TARGET_D']

print(len(training_set), len(test_set), len(data))
lin_reg = LinearRegression()

lin_reg.fit(train_features, training_label)

target_pred = lin_reg.predict(X=test_features)

print(len(target_pred), len(test_label))

calculations = []
for i in range(len(target_pred)):
    calculations.append(abs(test_label.tolist()[i] - target_pred[i]))
mae = np.mean(calculations)
print("Mean Absolute Error: ", mae)

standard_scalar = StandardScaler()


def lasso_kfold_cv(x, y, p, k):
    kframes = int(len(x) / k)
    train_error = []
    cv_error = []
    lasso = Lasso(alpha=p)
    for i in range(k):
        index_start = i * kframes
        index_end = x.shape[0] - kframes * (k - i - 1)
        train_y = y[index_start:index_end]
        train_x = x[index_start:index_end]
        validation_x = x.drop(x.index[index_start:index_end])
        validation_y = y.drop(y.index[index_start:index_end])
        lasso.fit(X=train_x, y=train_y)
        test_values = lasso.predict(validation_x)
        train_values = lasso.predict(train_x)
        train_error.append(np.mean(abs(train_values - train_y)))
        cv_error.append(np.mean(abs(test_values - validation_y)))

    lasso_predict(x, y, p)
    train_error = np.mean(train_error)
    cv_error = np.mean(cv_error)

    return train_error, cv_error


def ridge_kfold_cv(x, y, p, k):
    kframes = int(len(x) / k)
    train_error = []
    cv_error = []
    ridge = Ridge(alpha=p)

    for i in range(k):
        index_start = i * kframes
        index_end = x.shape[0] - kframes * (k - i - 1)
        train_y = y[index_start:index_end]
        train_x = x[index_start:index_end]
        validation_x = x.drop(x.index[index_start:index_end])
        validation_y = y.drop(y.index[index_start:index_end])
        ridge.fit(X=train_x, y=train_y)
        test_values = ridge.predict(validation_x)
        train_values = ridge.predict(train_x)
        train_error.append(np.mean(abs(train_values - train_y)))
        cv_error.append(np.mean(abs(test_values - validation_y)))

    ridge_predict(x, y, p)
    train_error = np.mean(train_error)
    cv_error = np.mean(cv_error)
    return train_error, cv_error


def lasso_predict(x, y, p):
    lasso = Lasso(alpha=p)
    lasso.fit(x, y)
    prediction = lasso.predict(test_features)

    prediction_error = np.mean(abs(prediction - test_label))
    print("MAE at lamda: 10^", p//10, prediction_error)


def ridge_predict(x, y, p):
    ridge = Ridge(alpha=p)
    ridge.fit(x, y)
    prediction = ridge.predict(test_features)

    prediction_error = np.mean(abs(prediction - test_label))
    print("MAE at lamda: 10^", p//10, prediction_error)


standard_data = data_expected

scalar = StandardScaler(with_mean=0, with_std=1)

standard_set = standard_data.drop('TARGET_D', axis=1, inplace=False)
standard_test = standard_data.loc[:, 'TARGET_D']
scalar_fit = scalar.fit_transform(standard_set)
standard_train = scalar_fit[:training_index]
standard_label = standard_test[:training_index]
ridge_lam = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


standard_fit = pd.DataFrame(standard_train, columns=standard_set.columns)

print(len(standard_fit), len(standard_label))
ridge_training_error = []
ridge_cross_validation_error = []
for i in ridge_lam:
    print("Ridge processing lamda of 10^", i)
    te, cve = ridge_kfold_cv(standard_fit, standard_label, 10**i, 5)
    ridge_training_error.append(te)
    ridge_cross_validation_error.append(cve)

plt.plot(ridge_lam, ridge_cross_validation_error, label="Cross Validation Error")
plt.plot(ridge_lam, ridge_training_error, label="Training Error")
plt.legend()
plt.show()

lasso_lam = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1, 1.25, 1.5, 1.75, 2]
lasso_training_error = []
lasso_validation_error = []

for i in lasso_lam:
    te, cve = lasso_kfold_cv(standard_fit, standard_label, 10**i, 5)
    lasso_training_error.append(te)
    lasso_validation_error.append(cve)