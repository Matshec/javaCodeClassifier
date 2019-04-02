import pandas
import os

class DataHarvester:

    CODE_FRAGMENT = 'code_fragment'

    def __init__(self, data_path):
        self.data_path = data_path
        self.read_data = pandas.DataFrame()

    def read_all_files(self):
        dir_content = os.listdir(self.data_path)
        for file in dir_content:
            path = os.path.join(self.data_path, file)
            self.read_data = self.read_data.append(pandas.read_csv(path))

    def read_file(self):
        self.read_data = pandas.read_csv(self.data_path)

    def cut_lines(self, max_len=100):
        self.read_data[self.CODE_FRAGMENT] = self.read_data[self.CODE_FRAGMENT].str[:max_len]

