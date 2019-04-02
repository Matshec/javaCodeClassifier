from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import tensorflow
import numpy as np
import re


class Encoder:
    LIKE_BIG_BIG_INT = 100000

    def __init__(self, data):
        self.data = data
        self.encoded = None
        self.encoded_label = None

        self.num_of_label_classes = None
        self.num_of_data_classes = None

        #split on white chars or literal .
        self.split_regex = re.compile("\s|\.")

        self.keywords_iterator = iter(range(2, self.LIKE_BIG_BIG_INT))

        # return 1 for unknown word - like variable or function NAME
        # this is the vocabulary
        # TODO might be good to change to encode just java words and leave rest to enocde with sklearn LabelEncoder
        self.java_keywords_mapping = defaultdict(lambda: 1, {
            'abstract': next(self.keywords_iterator),
            'assert': next(self.keywords_iterator),
            'boolean': next(self.keywords_iterator),
            'break': next(self.keywords_iterator),
            'byte': next(self.keywords_iterator),
            'case': next(self.keywords_iterator),
            'catch': next(self.keywords_iterator),
            'char': next(self.keywords_iterator),
            'class': next(self.keywords_iterator),
            'const': next(self.keywords_iterator),
            'continue': next(self.keywords_iterator),
            'default': next(self.keywords_iterator),
            'do': next(self.keywords_iterator),
            'double': next(self.keywords_iterator),
            'else': next(self.keywords_iterator),
            'enum': next(self.keywords_iterator),
            'extends': next(self.keywords_iterator),
            'final': next(self.keywords_iterator),
            'finally': next(self.keywords_iterator),
            'float': next(self.keywords_iterator),
            'for': next(self.keywords_iterator),
            'goto': next(self.keywords_iterator),
            'if': next(self.keywords_iterator),
            'implements': next(self.keywords_iterator),
            'import': next(self.keywords_iterator),
            'instanceof': next(self.keywords_iterator),
            'int': next(self.keywords_iterator),
            'interface': next(self.keywords_iterator),
            'long': next(self.keywords_iterator),
            'native': next(self.keywords_iterator),
            'new': next(self.keywords_iterator),
            'package': next(self.keywords_iterator),
            'private': next(self.keywords_iterator),
            'protected': next(self.keywords_iterator),
            'public': next(self.keywords_iterator),
            'return': next(self.keywords_iterator),
            'short': next(self.keywords_iterator),
            'static': next(self.keywords_iterator),
            'strictfp': next(self.keywords_iterator),
            'super': next(self.keywords_iterator),
            'switch': next(self.keywords_iterator),
            'synchronized': next(self.keywords_iterator),
            'this': next(self.keywords_iterator),
            'throw': next(self.keywords_iterator),
            'throws': next(self.keywords_iterator),
            'transient': next(self.keywords_iterator),
            'try': next(self.keywords_iterator),
            'void': next(self.keywords_iterator),
            'volatile': next(self.keywords_iterator),
            'while': next(self.keywords_iterator),
            'true': next(self.keywords_iterator),
            'false': next(self.keywords_iterator),
            'null': next(self.keywords_iterator),
            # also add ;
            ";": next(self.keywords_iterator),
        })

    def encode_data(self):
        """
        encode just code lines fragment
        first encode to vocabulary, then use  one-hot encoding
        :return:
        """

        def post_process_row(dt):
            """
            add post processing to row like one-hot encoding
            besically it takes preprocessed row and  where we have mapped the values from disctionary
            and processes it such that output per row is array where each position marks a certain word and value is num
            of occurences of that word
            :param dt: list row
            :return: np.array
            """
            tmp = label_encoder.transform(dt)
            tmp = to_categorical(tmp, num_classes=len(label_encoder.classes_))
            # for now sum matrix vector wise - add all columns as vectors
            # aggregate repeating words
            # each word has its's own posision and value is word count
            # TODO condsider different encodeing for exmple where words do not have position ( saved order) and value represents word mapping
            tmp = np.sum(tmp, axis=(0), keepdims=True)
            return tmp.flatten()

        encoded_keywords = self.data['code_fragment'].apply(self._encode_row)

        label_encoder = LabelEncoder()
        label_encoder.fit(list(self.java_keywords_mapping.values()))

        self.num_of_data_classes = len(label_encoder.classes_)
        #dot = lambda dt: np.sum(to_categorical(label_encoder.transform(dt), num_classes=len(label_encoder.classes_)), axis=(0), keepdims=True).flatten()
        encoded = encoded_keywords.apply(post_process_row)


        # create 2d array
        self.encoded = np.stack(encoded)

    def _encode_row(self, data):
        """
        Encode row using java keyword mapping dictionary
        :param data: string with java code
        :return: mapped list
        """
        # add ; literal as keyword
        repl = data.replace(';', ' ;')

        # split on white chars or . literal
        split = re.split(self.split_regex, repl)

        # remove empty
        non_empty = [elem for elem in split if elem]
        return [self.java_keywords_mapping[elem] for elem in non_empty]

    def encode_label(self):
        """
        encode labels using one-hot encoding
        """
        label_encoder = LabelEncoder()
        y = self.data['label']
        label_encoder.fit(y)
        self.num_of_label_classes = len(label_encoder.classes_)
        self.encoded_label = to_categorical(label_encoder.transform(y))


