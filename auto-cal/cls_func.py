import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def prepare_traindata(filename: str, location: str = None, model_formula: str = 'cvrmse ~ .', train_data_sampling: bool = True, train_data_number: int = 0):
    """
    Prepares the training data for the model based on the given parameters.
    :param filename: The name of the CSV file containing the data.
    :param location: The location of the CSV file. Default is None.
    :param model_formula: The formula specifying the model structure. Default is 'cvrmse ~ .'.
    :param train_data_sampling: A boolean indicating whether to sample the training data. Default is True.
    :param train_data_number: The number of samples to extract from the training data. Default is 0.
    :return data_dict: A dictionary containing the original data, training data, test data, model response, and model explanatory variables.
    """
    if location != None:
        data_total = pd.read_csv(location + filename + '.csv')
    else:
        data_total = pd.read_csv(filename + '.csv')

    data_columns = data_total.columns.values.tolist()
    if 'Unnamed: 0' in data_columns:
        data_total.rename(columns = {'Unnamed: 0' : 'observations'}, inplace = True)

    data_columns = data_total.columns.values.tolist()

    model_response = [model_formula.split('~')[0].strip()]
    model_explanat = []
    if model_formula.split('~')[1].strip() != '.':
        model_explanat = [e.strip() for e in model_formula.split('~')[1].split('+')]
    else:
        model_explanat = [col for col in data_columns if col != 'observations' and col != model_response[0]]

    model_variable = model_response + model_explanat
    for var in model_variable:
        if var not in data_columns:
            ValueError("Check the variables or formula")

    y_var = model_response
    x_var = [var for var in model_variable if var not in y_var]

    X_data = data_total[x_var].values
    y_data = data_total[y_var].values

    train_samples = data_total.sample(n=len(data_total))
    if train_data_sampling == True:
        train_samples = data_total.sample(n=train_data_number)

    test_samples = data_total.loc[~data_total.index.isin(train_samples.index)]

    X_train = train_samples[x_var].values
    y_train = train_samples[y_var].values

    X_test = test_samples[x_var].values
    y_test = test_samples[y_var].values

    original_data_set = [X_data, y_data]
    train_data_set = [X_train, y_train]
    test_data_set = [X_test, y_test]

    variable = [model_response, model_explanat]

    return {'original_data': original_data_set, 'train_data': train_data_set, 'test_data':test_data_set, 'model_response': model_response, 'model_explanat':model_explanat}

class Kriging_rbf:
    """
    Gaussian Process Regression / Radial basis function(RBF) algorithm used for kernel.
    Warning: length_scale of the RBF kernel function must be customed by the users.
            Recommended to use optimization method to find the optimal value for length_scale
            i.e. scipy.optimize.minimize"""
    def __init__(self, train_data_set, test_data_set, result_to_csv:bool = True, **kwargs):
        """
        Initializes the Kriging_rbf class.
        Parameters:
        :param train_data_set: A list containing the training input data (train_X) and target data (train_y).
        :param test_data_set: A list containing the test input data (test_X) and an empty list for test_y.
        :param result_to_csv: A boolean indicating whether to write the result to a CSV file. Default is True.
        :param **kwargs: Additional keyword arguments. In this case, 'length_scale' can be passed to customize the RBF kernel length_scale. """
        self.train_data_set = train_data_set
        self.test_data_set = test_data_set

        kernel = RBF()
        if 'length_scale' in kwargs:
            length_scale = kwargs['length_scale']
            kernel = RBF(length_scale=length_scale)
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(train_data_set[0], train_data_set[1])
        self.fitted_gpr = gpr

        if len(test_data_set) == 0:
            test_data_set_arg = train_data_set[0]
            self.predicted_data = self.predict(test_data_set_arg)
            if result_to_csv == True:
                train_data_df=pd.DataFrame(train_data_set[0])
                self._write_result_to_csv(original_dataframe=train_data_df, result_data=self.predicted_data, filename='result', location=None)
        else:
            test_data_set_arg = test_data_set[0]
            self.predicted_data = self.predict(test_data_set_arg)
            if result_to_csv == True:
                test_data_df = pd.DataFrame(test_data_set[0])
                self._write_result_to_csv(original_dataframe=test_data_df, result_data=self.predicted_data, filename='result', location=None)

    def predict(self, test_X_data):
        """
        Predicts the target values for the given test input data using the fitted GPR model.
        :param test_X_data: The input data for which the target values are predicted.
        :return y_pred: The predicted target values."""
        y_pred = self.fitted_gpr.predict(test_X_data)
        return y_pred

    def _write_result_to_csv(self, original_dataframe, result_data, filename:str = 'result', location:str = None):
        """
        Writes the result data along with the original input data to a CSV file.
        :param original_dataframe: The original input data as a DataFrame.
        :param result_data: The predicted target values.
        :param filename: The name of the output CSV file. Default is 'result'.
        :param location: The location to save the CSV file. Default is None."""
        result_df = original_dataframe.copy()
        result_df['GPR-y'] = result_data
        if location == None:
            result_df.to_csv(filename + '.csv', index=False)
        else:
            result_df.to_csv(location + filename + '.csv', index = False)

class L_BFGS_B:
    def __init__(self, input_data: list, surrogate_model):
        """
        Initializes the class by assigning input and target data and performs an optimization process to determine the optimal input variables and target prediction.
        :param input_data: A list containing input data, where the first element is the input variables (X_data) and the second element is the target variables (y_data).
        :param surrogate_model: A surrogate model used for prediction."""
        X_data, y_data = input_data[0], input_data[1]
        self.surrogate_model = surrogate_model
        self.X_data = X_data
        self.y_data = y_data

        results = self.optimize_inputs(self.X_data)
        self.optimize_result = {'optimal_explanat': results[0], 'optimal_response': results[1]}

    def optimize_inputs(self, input_X_data):
        """
        optimizes input variables using the L-BFGS-B method and returns the optimal inputs and corresponding target prediction.
        :param input_X_data: Input variables (X_data) used for optimization.
        :return list: A list containing the optimal input variables and the corresponding target prediction. """
        bounds = [(np.min(input_X_data[:, i]), np.max(input_X_data[:, i])) for i in range(input_X_data.shape[1])]
        result = minimize(self.objective_function, x0=np.mean(input_X_data, axis=0), method='L-BFGS-B', bounds=bounds)
        optimal_inputs = result.x
        optimal_target = self.objective_function(optimal_inputs)

        return [optimal_inputs, optimal_target]

    def objective_function(self, explanatory_variable_combi):
        """
        an objective function that calculates the prediction of the target variable based on a given combination of explanatory variables using a surrogate model.
        :param explanatory_variable_combi: A combination of explanatory variables for which the target variable prediction is calculated.
        :return ndarray: The prediction of the target variable. """
        explanatory_variable_combi = np.array(explanatory_variable_combi).reshape(1, -1)  # x to 2D array format
        self.explanatory_variable_combi = explanatory_variable_combi
        combi_prediction = self.surrogate_model.predict(self.explanatory_variable_combi)  # target var- prediction by explanatory variable combi

        return combi_prediction



