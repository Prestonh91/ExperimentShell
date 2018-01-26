import pandas as pd
import numpy as np


class FileReader:
    def __init__(self):
        pass

    def get_header(self, file_name):
        header = []
        if file_name == 'car.txt':
            header = ["Buying_price", "Maint_costs", "Doors", "Seats",
                      "Luggage_cap", "Safety_rate", "Acceptability_rate"]
        elif file_name == 'mgp.txt':
            header = ['MPG', 'Cylinders', 'Displacement', 'HP', 'Weight',
                      'Accel', 'Model', 'Origin', 'Car_Name']
        elif file_name == 'health.txt':
            header = ["Pregnancies", "Glucose_Tol", "Diastolic",
                      "Tricep", "Insulin", "BMI", "Pedigree",
                      "Age", "Diabetes"]
        return header

    def read_file(self, file_name):
        headers = []
        if file_name == "car.txt":
            headers = self.get_header(file_name)
        elif file_name == "mpg.txt":
            headers = self.get_header(file_name)
        elif file_name == "health.txt":
            headers = self.get_header(file_name)

        if file_name == "car.txt" or file_name == "health.txt":
            data = pd.read_csv(file_name, header=None, delimiter=",",
                               names=headers, skipinitialspace=True)
        else:
            data = pd.read_csv(file_name, header=None, delimiter=",",
                               skipinitialspace=True, engine='python')
            data = np.array(data)

        return data
