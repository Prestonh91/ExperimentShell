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