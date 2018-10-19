import pandas as pd
import numpy as np
import sklearn.preprocessing as sk


# load and encode categoriacal data
# PARAM: filepath - file-path to the CSV file.
# RETURN: data - A dataframe object containing encoded data.
def load(filepath):
    raw_data = pd.read_csv(filepath)
    df = pd.DataFrame({})
    new_data = []
    for column in raw_data:
        if raw_data[column].dtype == 'object' or raw_data[column].nunique() < 20:
            new_data.append(column)
    raw_data = pd.get_dummies(raw_data, columns=new_data, prefix=new_data)

    return raw_data
