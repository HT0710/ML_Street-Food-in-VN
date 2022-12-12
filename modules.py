from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import pandas as pd


class DATA_PROCESSING:
    def __init__(self, dataset):
        self.__raw_data = pd.read_csv(dataset)
        self.__prepared_data = self.__prepare()
        self.__result = None

    def __prepare(self):
        data = self.__raw_data.values

        container = []
        for i, o in enumerate(data):
            item = o.tolist()[3:]
            cell = [[] for x in item] if (i % 3 == 0) else cell

            for j, p in enumerate(item):
                cell[j].append(p)

            if i % 3 == 2:
                container.append(cell)

        return container

    def get_dataset(self):
        return self.__prepared_data

    def get_result(self):
        return self.__result

    def mean(self):
        data = self.__prepared_data
        for i, o in enumerate(data):
            for j, p in enumerate(o):
                data[i][j] = np.mean(p)

        self.__result = data
        return self.__result

    def median(self):
        data = self.__prepared_data
        for i, o in enumerate(data):
            for j, p in enumerate(o):
                data[i][j] = np.median(p)

        self.__result = data
        return self.__result

    def mdmi(self):
        """Median Dependant Mean Independent"""
        data = self.__prepared_data
        for i, o in enumerate(data):
            for j, p in enumerate(o):
                if j == 0:
                    data[i][0] = np.median(p)
                    continue
                data[i][j] = np.mean(p)

        self.__result = data
        return self.__result


class METHODS:
    @staticmethod
    def binarize(dataset, index_of_dependent_variable):
        pd.options.mode.chained_assignment = None
        for i, o in enumerate(dataset.iloc[:, index_of_dependent_variable]):
            if o >= 2.5:
                dataset.iloc[:, index_of_dependent_variable][i] = 1
            else:
                dataset.iloc[:, index_of_dependent_variable][i] = 0
        return dataset

    @staticmethod
    def upsampling(dataset, index_of_dependent_variable: int):
        zero_data = dataset[dataset[index_of_dependent_variable] == 0]
        one_data = dataset[dataset[index_of_dependent_variable] == 1]

        zero_data = resample(zero_data, replace=True, n_samples=len(one_data), random_state=2)

        new_dataset = pd.concat([zero_data, one_data], axis=0)
        return new_dataset

    @staticmethod
    def scaler_list():
        return [None, StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), Normalizer()]

    @staticmethod
    def scaling(train_data, test_data, scaler):
        if scaler is not None:
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)
        return train_data, test_data

    @staticmethod
    def mean(data) -> None:
        print("Best: ", np.max(data))
        print("Worst: ", np.min(data))
        print("Mean: ", np.mean(data))

    def run(self, dataset, algorithm, scaler, test_size=0.2):
        features = dataset.iloc[:, 1:]
        labels = dataset.iloc[:, 0]
        # print(labels.shape, Counter(labels))

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
        # print("Test size: ", y_test.shape)

        X_train, X_test = self.scaling(X_train, X_test, scaler)

        algorithm.fit(X_train, y_train)
        y_pred = algorithm.predict(X_test)

        return y_test, y_pred
