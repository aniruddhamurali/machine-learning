from math import exp

class LogisticRegression:

    def __init__(self, n_folds, l_rate, n_epoch):
        self.n_folds = n_folds
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        
                 
    # Find the min and max values for each column
    def dataset_minmax(self, dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax


    # Rescale dataset columns to the range 0-1
    def normalize_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


    # Split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split


    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


    # Evaluate an algorithm using a cross validation split
    def evaluate(self, dataset, algorithm, n_folds, *args):
        folds = cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores


    # Make a prediction with coefficients
    def predict_coef(self, row, coefficients):
        yhat = coefficients[0]
        for i in range(len(row)-1):
            yhat += coefficients[i + 1] * row[i]
        return 1.0 / (1.0 + exp(-yhat))


    # Estimate logistic regression coefficients using stochastic gradient descent
    def coefficients_sgd(self, train, l_rate, n_epoch):
        coef = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            for row in train:
                yhat = predict(row, coef)
                error = row[-1] - yhat
                coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
                for i in range(len(row)-1):
                    coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
        return coef


    # Logistic Regression Algorithm With Stochastic Gradient Descent
    def predict(self, train, test, l_rate, n_epoch):
        predictions = list()
        coef = coefficients_sgd(train, l_rate, n_epoch)
        for row in test:
            yhat = predict_coef(row, coef)
            print(yhat)
            yhat = round(yhat)
            predictions.append(yhat)
        return(predictions)
