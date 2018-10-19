import loaddata_lab5_mai_andrew as loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

training_percentage = 0.3

data = loader.load("C:\\Users\\Luyen\\OneDrive\\Documents\\BCIT_WORK\\Machine Learning\\lab5\\data_lab5.csv")
data_expected = pd.read_csv(
    "C:\\Users\\Luyen\\OneDrive\\Documents\\BCIT_WORK\\Machine Learning\\lab5\\data_lab5_expanded.csv")

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

target_pred = lin_reg.predict(test_features)

print(len(target_pred), len(test_label))

calculations = []
for i in range(len(target_pred)):
    calculations.append(abs(test_label.tolist()[i] - target_pred[i]))
mae = np.mean(calculations)
print("Mean Absolute Error: ", mae)

standard_scalar = StandardScaler()


def poly_kfold_cv(x, y, p, k):
    kframes = int(len(x) / k)
    train_error = []
    cv_error = []
    for i in range(k):
        index_start = i * kframes
        index_end = x.shape[0] - kframes * (k - i - 1)
        train_y = y[index_start:index_end]
        train_x = x[index_start:index_end]
        validation_x = x.drop(x.index[index_start:index_end])
        validation_y = y.drop(y.index[index_start:index_end])
        ridge = Ridge(alpha=p)
        ridge.fit(train_x, train_y)
        test_values = ridge.predict(validation_x)
        train_values = ridge.predict(train_x)

        train_error.append(np.mean(abs(train_values - train_y)))
        cv_error.append(np.mean(abs(test_values - validation_y)))
    print(cv_error)
    return np.mean(train_error), np.mean(cv_error)


standard_data = data
standard_set = data.drop('TARGET_D', axis=1, inplace=False)
standard_test = data.loc[:, 'TARGET_D']
std_scalar = StandardScaler()
standard_fit = std_scalar.fit_transform(standard_set)

standard_train = standard_fit[:training_index]
standard_test_set = standard_fit[training_index + 1:]
standard_label = standard_test[:training_index]
lam = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

standard_train = pd.DataFrame(standard_train, columns=train_features.columns)
print(len(standard_train), len(standard_test))
train_error = []
cv_error = []
for i in lam:
    te, cve = poly_kfold_cv(standard_train, standard_label, 10**i, 5)
    train_error.append(te)
    cv_error.append(cve)
plt.plot(lam, cv_error, label="Cross Validation Error")
plt.plot(lam, train_error, label="Training Error")
plt.legend()
plt.show()
