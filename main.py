import numpy as np
from argparse import ArgumentParser
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from encoder import Encoder
from dataHarverster import DataHarvester
from model import ModelBuilder


def args():
    """
    get path to data as arg
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("path", help="path to .csv data file")
    return parser.parse_args()

if __name__ == '__main__':
    args = args()
    DATA_PATH = args.path

    seed = 42
    np.random.seed(seed)

    harvester = DataHarvester(DATA_PATH)
    harvester.read_file()
    harvester.cut_lines()

    encoder = Encoder(harvester.read_data)
    encoder.encode_data()
    encoder.encode_label()

    X = encoder.encoded
    Y = encoder.encoded_label

    model_builder = ModelBuilder(encoder.num_of_label_classes, encoder.num_of_data_classes)
    estimator = KerasClassifier(build_fn=model_builder, epochs=20, batch_size=5, verbose=5)
    kfold = KFold(n_splits=30, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
