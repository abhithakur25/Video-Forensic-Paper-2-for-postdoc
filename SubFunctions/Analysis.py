import os

import numpy as np
from termcolor import cprint

from SubFunctions.Evaluate import Evaluation_Metrics, Evaluation_Metrics1
from SubFunctions.Model import Network
# from SubFunctions.Evaluate import Evaluation_Metrics
from sklearn.model_selection import KFold


def train_test_split(data, train_size):

    labels = data['labels']
    # Get the unique classes in the target variable 'y'
    num_classes = np.unique(labels)

    # Initialize empty lists to store training and testing data
    x_train1 = []

    y_train = []

    x_test1 = []

    y_test = []

    # Loop through each unique class
    for i in range(len(num_classes)):
        # Find indices of samples belonging to the current class
        indices = np.where(labels == num_classes[i])

        # Split the indices based on the specified 'train_size'
        train_index = indices[0][:int(len(indices[0]) * train_size)]
        test_index = indices[0][int(len(indices[0]) * train_size):]

        # Extract features and labels for training set
        train_feat1 = data['features'][train_index]

        train_lab = labels[train_index]

        # Extract features and labels for testing set
        test_feat1 = data['features'][test_index]

        test_lab = labels[test_index]

        # Extend the lists with the current class data
        x_train1.extend(train_feat1)


        y_train.extend(train_lab)

        x_test1.extend(test_feat1)


        y_test.extend(test_lab)

    # Convert the lists to numpy arrays
    x_train1 = np.array(x_train1)


    y_train = np.array(y_train)

    x_test1 = np.array(x_test1)

    y_test = np.array(y_test)

    train_samples = x_train1.shape[0]
    train_indices = np.random.permutation(train_samples)

    x_trainC1 = x_train1[train_indices]

    y_train = y_train[train_indices]

    test_samples = x_test1.shape[0]
    test_indices = np.random.permutation(test_samples)

    x_testC1 = x_test1[test_indices]

    y_test = y_test[test_indices]

    # Separate features and labels after shuffling
    return [x_trainC1,
            x_testC1,
            y_train.astype(int), y_test.astype(int)]




class TPAnalysis:

    def __init__(self, data):
        """
        Initialize the Analysis class.

        Args:
        - Features: The feature data for analysis.
        - Labels: The labels corresponding to the feature data.
        """
        self.data = data
        self.epochs = 500
        self.perf_epochs = [100, 200, 300, 400, 500]

    def ComparativeAnalysis(self):
        """
        Perform Comparative Analysis to compare the proposed method with existing methods.

        Vary the training percentage and use different classification methods.

        Save the results in numpy files for each method and training percentage.
        """
        # Initialize lists to store comparative analysis results
        ComparativeResults = []

        TrainingPercentage = 0.4

        for i in range(6):
            cprint(f"[⚠️] Comparative Analysis Count Is {i} Out Of 6", 'cyan', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage
            data = train_test_split(self.data, train_size=TrainingPercentage)

            params = {'x_train': data[0],
                      'x_test': data[1],
                      'y_train': data[2], 'y_test': data[3], 'epochs': self.epochs}



            Ne = Network(**params)

            # Perform cl classification using different methods and get predictions
            output = [
                Ne.BiLSTMGBM()]

            # Calculating the Performance
            ComparativeResults.append([Evaluation_Metrics(data[3], y_pred) for y_pred in output])
            # Increase the training percentage for the next iteration
            TrainingPercentage += 0.1

        perf_names = ['A']
        # , 'L', 'M'
        file_names = [f'Analysis1\\TP\\COM_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in ComparativeResults])
        cprint("[✅] Execution of Comparative Analysis Completed ", 'green', on_color='on_grey')


    def PerformanceAnalysis(self):
        """
        Perform Performance Analysis to check the maximum performance of the proposed method.

        Vary the training percentage and epochs.

        Save the results in numpy files for each training percentage and epoch combination.
        """
        # Initialize lists to store performance analysis results
        PerformanceResults = []

        TrainingPercentage = 0.4

        for i in range(6):
            cprint(f"[⚠️] Performance Analysis Count Is {i} Out Of 6", 'cyan', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage
            data = train_test_split(self.data, train_size=TrainingPercentage)
            params = {'x_train': data[0],
                      'x_test': data[1],
                      'y_train': data[2], 'y_test': data[3], 'epochs': self.epochs}

            Ne = Network(**params)

            # Perform cl classification using different methods and get predictions
            output = [
                Ne.BiLSTMGBM(epochs=self.perf_epochs[0]),
                Ne.BiLSTMGBM(epochs=self.perf_epochs[1]),
                Ne.BiLSTMGBM(epochs=self.perf_epochs[2]),
                Ne.BiLSTMGBM(epochs=self.perf_epochs[3]),
                Ne.BiLSTMGBM(epochs=self.perf_epochs[4])]

            # Calculating the Performance
            PerformanceResults.append([Evaluation_Metrics(data[3], y_pred) for y_pred in output])
            # Increase the training percentage for the next iteration
            TrainingPercentage += 0.1

        perf_names = ['A', 'B', 'C', 'D', 'E']
        # , 'L', 'M'
        file_names = [f'Analysis1\\TP\\PERF_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in PerformanceResults])


        # Print a completion message
        cprint("[✅] Execution of Performance Analysis Completed ", 'green', on_color='on_grey')


    def RocAnalysis(self):
        cprint("[INFO] Executing Analysis", 'grey', on_color='on_white')

        FPR = []
        TPR = []

        # Define a list of training set percentages.
        Tr_Per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for i in range(len(Tr_Per)):
            cprint("Analysis Count Is {0} Out Of 9".format(i + 1), 'blue', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage
            data = train_test_split(self.data, train_size=Tr_Per[i])

            params = {'x_train': data[0],
                      'x_test': data[1],
                      'y_train': data[2], 'y_test': data[3], 'epochs': self.epochs}


            Ne = Network(**params)

            # Perform cl classification using different methods and get predictions

            # Perform cl classification using different methods and get predictions
            pred_c1 = Ne.BiLSTMGBM()



            # Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for each classifier.
            [TPR1c, FPR1c] = Evaluation_Metrics1(data[3], pred_c1)


            # Store FPR and TPR for all classifiers.
            FPR_all = [FPR1c]
            TPR_all = [TPR1c]

            FPR.append(FPR_all)
            TPR.append(TPR_all)

        np.save(f'{os.getcwd()}\\Analysis1\\TP\\TPR.npy', TPR)
        np.save(f'{os.getcwd()}\\Analysis1\\TP\\FPR.npy', FPR)


        cprint("[INFO] Analysis Completed", 'green', on_color='on_grey')




class KFAnalysis:
    """
    K-fold analysis, often referred to as k-fold cross-validation, is a common technique used in machine learning
    and statistics to assess the performance and robustness of a predictive model. It is particularly useful when
    you have a limited amount of data and want to ensure that your model is not overfitting (performing well on
    training data but poorly on new, unseen data).
    """

    def __init__(self, data):
        """
        Initialize the Analysis class.

        Args:
        - Features: The feature data for analysis.
        - Labels: The labels corresponding to the feature data.
        """
        self.data = data
        self.epochs = 500
        self.folds = [6, 7, 8, 9, 10]
        self.perf_epochs = [100, 200, 300, 400, 500]


    @staticmethod
    def train_test_split(train, test, data):
        # A static method to extract training and testing data based on indices
        x_train1 = []


        y_train = []

        x_test1 = []

        y_test = []

        for i in range(len(data['features'])):
            if i in train:
                x_train1.append(data['features'][i])

                y_train.append(data['labels'][i])
            else:

                x_test1.append(data['features'][i])
                y_test.append(data['labels'][i])

        x_train1 = np.array(x_train1)

        x_test1 = np.array(x_test1)


        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Separate features and labels after shuffling
        return [x_train1,
                x_test1,
                y_train.astype(int), y_test.astype(int)]

    def ComparativeAnalysis(self):
        # Perform Comparative Analysis
        ComparativeResults_all = []

        for i in range(len(self.folds)):
            # Iterate through different fold values
            # Iterate through different fold values
            cprint(f"[⚠️] No.of Fold : {self.folds[i]} ", 'cyan', on_color='on_grey')

            k_fold = KFold(n_splits=self.folds[i], random_state=1, shuffle=True)

            ComparativeResults = []

            for j, [train, test] in enumerate(k_fold.split(self.data['features'])):

                # Iterate through K-fold splits
                data = self.train_test_split(train, test, self.data)

                params = {'x_train': data[0],
                          'x_test': data[1],
                          'y_train': data[2], 'y_test': data[3], 'epochs': self.epochs}

                Ne = Network(**params)

                # Perform cl classification using different methods and get predictions
                output = [
                    Ne.BiLSTMGBM()]
                # Calculating the Performance
                ComparativeResults.append([Evaluation_Metrics(data[3], y_pred) for y_pred in output])

                # Increase the training percentage for the next iteration

                # Compute the mean of performance metrics for each method and fold
            ComparativeResults_all.append(np.mean(np.array(ComparativeResults), axis=0))

            # Save the results as numpy arrays

        perf_names = ['A']

        file_names = [f'Analysis1\\KF\\COM_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in ComparativeResults_all])

        cprint("[✅] Execution of Comparative Analysis Completed ", 'green', on_color='on_grey')

    def PerformanceAnalysis(self):
        # Perform Comparative Analysis
        PerformanceResults_all = []

        for i in range(len(self.folds)):
            # Iterate through different fold values
            cprint(f"[⚠️] No.of Fold : {self.folds[i]} ", 'cyan', on_color='on_grey')

            k_fold = KFold(n_splits=self.folds[i], random_state=1, shuffle=True)

            PerformanceResults = []

            for j, [train, test] in enumerate(k_fold.split(self.data['proposed'])):
                # Iterate through K-fold splits
                data = self.train_test_split(train, test, self.data)

                params = {'x_train': data[0],
                          'x_test': data[1],
                          'y_train': data[2], 'y_test': data[3], 'epochs': self.epochs}

                Ne = Network(**params)

                # Perform cl classification using different methods and get predictions
                output = [
                    Ne.BiLSTMGBM(epochs=self.perf_epochs[0]),
                    Ne.BiLSTMGBM(epochs=self.perf_epochs[1]),
                    Ne.BiLSTMGBM(epochs=self.perf_epochs[2]),
                    Ne.BiLSTMGBM(epochs=self.perf_epochs[3]),
                    Ne.BiLSTMGBM(epochs=self.perf_epochs[4])]
                # Calculating the Performance
                PerformanceResults.append([Evaluation_Metrics(data[3], y_pred) for y_pred in output])

                # Increase the training percentage for the next iteration

                # Compute the mean of performance metrics for each method and fold
            PerformanceResults_all.append(np.mean(np.array(PerformanceResults), axis=0))

        # Save the results as numpy arrays
        perf_names = ['A', 'B', 'C', 'D', 'E']
        # , 'L', 'M'
        file_names = [f'Analysis1\\KF\\PERF_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in PerformanceResults_all])
        # Print a completion message
        cprint("[✅] Execution of Performance Analysis Completed ", 'green', on_color='on_grey')

