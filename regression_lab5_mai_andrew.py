import loaddata_lab5_mai_andrew as loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import itertools
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
# from multiprocessing.dummy import Pool as tp

training_percentage = 0.3

data = loader.load("data_lab5.csv")
data_expected = pd.read_csv(
    "data_lab5_expanded.csv")

print(len(data), len(data_expected))

training_index = int(len(data) * training_percentage)

training_set = data.loc[:training_index]
train_features = training_set.drop('TARGET_D', axis=1, inplace=False)
training_label = training_set.loc[:, 'TARGET_D']

test_set = data.loc[training_index + 1:]
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


# Tests kasso and kfold based on the lamda coming in
# PARAM x the sample values.
# PARAM y the prediction values
# PARAM p lambda entering by magnitude 10^p
# PARAM k k folds, 5 by default
def lasso_kfold_cv(x, y, p, k):
    kframes = int(len(x) / k)
    train_error = []
    cv_error = []
    lasso = Lasso(alpha=p, max_iter=10e5)
    for i in range(k):
        index_start = i * kframes
        index_end = x.shape[0] - kframes * (k - i - 1)
        validation_y = y[index_start:index_end]
        validation_x = x[index_start:index_end]
        train_x = x.drop(x.index[index_start:index_end])
        train_y = y.drop(y.index[index_start:index_end])
        lasso.fit(X=train_x, y=train_y)
        test_values = lasso.predict(validation_x)
        train_values = lasso.predict(train_x)
        train_error.append(np.mean(abs(train_values - train_y)))
        cv_error.append(np.mean(abs(test_values - validation_y)))

    mean_error = lasso_predict(x, y, p)
    train_error = np.mean(train_error)
    cv_error = np.mean(cv_error)
    return train_error, cv_error, mean_error


# Tests ridge regression and kfold based on the lamda coming in
# PARAM x the sample values.
# PARAM y the prediction values
# PARAM p lambda entering by magnitude 10^p
# PARAM k k folds, 5 by default
def ridge_kfold_cv(x, y, p, k):
    kframes = int(len(x) / k)
    train_error = []
    cv_error = []
    ridge = Ridge(alpha=p)

    for i in range(k):
        index_start = i * kframes
        index_end = x.shape[0] - kframes * (k - i - 1)
        validation_y = y[index_start:index_end]
        validation_x = x[index_start:index_end]
        train_x = x.drop(x.index[index_start:index_end])
        train_y = y.drop(y.index[index_start:index_end])
        ridge.fit(X=train_x, y=train_y)
        test_values = ridge.predict(validation_x)
        train_values = ridge.predict(train_x)
        train_error.append(np.mean(abs(train_values - train_y)))
        cv_error.append(np.mean(abs(test_values - validation_y)))

    mean_error = ridge_predict(x, y, p)
    train_error = np.mean(train_error)
    cv_error = np.mean(cv_error)
    return train_error, cv_error, mean_error


# Predits with lasso using test and training sets
def lasso_predict(x, y, p):
    lasso = Lasso(alpha=p)
    lasso.fit(x, y)
    prediction = lasso.predict(test_features)
    prediction_error = np.mean(abs(prediction - test_label))
    print("MAE of Lasso prediction at lamda", p, ":", prediction_error)
    return prediction_error


# Predits with ridge using test and training sets
def ridge_predict(x, y, p):
    ridge = Ridge(alpha=p)
    ridge.fit(x, y)
    prediction = ridge.predict(test_features)

    prediction_error = np.mean(abs(prediction - test_label))
    print("MAE of Ridge prediction at lamda", p, ":", prediction_error)
    return prediction_error

# Data standarization
standard_data = data

scalar = StandardScaler()

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
lowest_ridge_mae = 0
ideal_ridge_lamda = 0
for i in ridge_lam:
    print("Ridge processing lamda of 10^", i)
    te, cve, mean = ridge_kfold_cv(standard_fit, standard_label, 10**i, 5)
    ridge_training_error.append(te)
    ridge_cross_validation_error.append(cve)
    if lowest_ridge_mae == 0 or lowest_ridge_mae > mean:
        lowest_ridge_mae = mean
        ideal_ridge_lamda = i

plt.plot(ridge_lam, ridge_cross_validation_error, label="Cross Validation Error")
plt.plot(ridge_lam, ridge_training_error, label="Training Error")
plt.title("Ridge Regression Error Graph")
plt.xlabel("Lambda by power of 10's")
plt.ylabel("Mean error")
plt.legend()
plt.show()

lasso_lam = [10**-2, 10**-1.75, 10**-1.5, 10**-1.25, 10**-1,
             10**-0.75, 10**-0.50, 10**-0.25, 10**0, 10**0.25, 10**0.50, 10**0.75,
             10**1, 10**1.25, 10**1.5, 10**1.75, 10**2]
lasso_training_error = []
lasso_validation_error = []
lowest_lasso_mae = 0
ideal_lasso_lamda = 0


for i in lasso_lam:
    print("Lasso processing lamda of 10^", i)
    te, cve, mean = lasso_kfold_cv(standard_fit, standard_label, i, 5)
    lasso_training_error.append(te)
    lasso_validation_error.append(cve)
    if lowest_lasso_mae == 0 or lowest_lasso_mae > mean:
        lowest_lasso_mae = mean
        ideal_lasso_lamda = i
plt.plot(lasso_lam, lasso_validation_error, label="Cross Validation Error")
plt.plot(lasso_lam, lasso_training_error, label="Training error")
plt.xlabel("Lambda by power of 10's")
plt.ylabel("Mean error")
plt.title("Lasso Error Table")
plt.legend()
plt.show()


print("Linear Regression MAE: ", mae,
      "\nLowest Ridge Regression MAE:", lowest_ridge_mae, "at Lambda 10^", ideal_ridge_lamda,
      "\nLowest Lasso Regression MAE:", lowest_lasso_mae, "at Lambda 10^", ideal_lasso_lamda)
print("")
