from sklearn import preprocessing
from copy import deepcopy
import pandas as pd
import numpy as np


class DataProcessor:
    def __init__(self):
        pass

    def process_cars1(self, data):
        clean_up = {'Buying_price': {'vhigh': 6, 'high': 4, 'med': 2, 'low': 0},
                    'Maint_costs': {'vhigh': 6, 'high': 4, 'med': 2, 'low': 0},
                    'Doors': {'2': 2, '3': 3, '4':4, '5more': 5},
                    'Seats': {'2':2, '4':4, 'more': 6},
                    'Luggage_cap': {'big': 6, 'med': 3, 'small': 1},
                    'Safety_rate': {'high':6, 'med': 3, 'low': 1},
                    'Acceptability_rate': {'vgood': 6, 'good': 4, 'acc': 2,
                                           'unacc': 0}}
        data.replace(clean_up, inplace=True)
        targets = data[[6]]
        new_data = data.drop("Acceptability_rate", axis=1)

        np_data = np.array(new_data.values.tolist())
        np_targets = np.array(targets.values.tolist())

        return np_data, np_targets

    def process_health(self, data):
        np.set_printoptions(suppress=True)

        # Replace "bad" zeros
        data[["Glucose_Tol", "Diastolic", "Tricep", "Insulin", "BMI"]] = \
            data[[1,2,3,4,5]].replace(0, np.NaN)
        data.fillna(data.mean(), inplace=True)

        #Remove targets before Standardizing and Normalizing
        targets = data[[8]]
        data = data.drop("Diabetes", axis=1)

        # Declare Standardization and Min_Max, scale glucose test
        scalar = preprocessing.StandardScaler(with_mean=False)
        data_norm = preprocessing.minmax_scale(data)
        data = scalar.fit_transform(data)

        data_norm = np.array(data_norm)
        targets = np.array(targets)
        data = np.array(data)

        return data, data_norm, targets