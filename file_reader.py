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
        elif file_name == 'car2.txt':
            header = []
        elif file_name == 'health.txt':
            header = []

        return header

    def read_file(self, file_name):
        headers = []
        if file_name == "car.txt":
            headers = self.get_header(file_name)
        elif file_name == "car2.txt":
            headers = self.get_header(file_name)
        elif file_name == "health.txt":
            headers = self.get_header(file_name)

        data = pd.read_csv(file_name, header=None, delimiter=",", names=headers,
                           skipinitialspace=True)
        return data