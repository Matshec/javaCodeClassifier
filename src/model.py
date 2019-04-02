from keras import Sequential
from keras.layers import Dense


class ModelBuilder:

    def __init__(self, label_classes, data_classes):
        self.label_classes = label_classes
        self.data_classes = data_classes

    def __call__(self, *args, **kwargs):
        """
        make object of class callable for purpose of Keras Classifier
        :param args:
        :param kwargs:
        :return:
        """
        return self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.data_classes, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.label_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
