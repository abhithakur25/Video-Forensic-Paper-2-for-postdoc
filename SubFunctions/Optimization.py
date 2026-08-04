import numpy as np
from termcolor import cprint
from SubFunctions.Evaluate import Evaluation_Metrics
from mealpy import FloatVar
from mealpy.Proposed import HYBRID







class Optimization:
    """
    Optimization is optimizing the trained model weights for better performance.
    """

    def __init__(self, model, x_test, y_test):
        """
        Initialize the Optimization class.

        Args:
        - model: The trained neural network model.
        - x_test: Test data used for evaluation.
        - y_test: Ground truth labels for the test data.
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def fitness_function1(self, solution):
        """
        Calculate the fitness value of a solution.

        Args:
        - solution: The solution to be evaluated.

        Returns:
        - Pre: A performance metric value.
        """
        # Get the current model weights
        to_opt = self.model.get_weights()

        to_opt_1 = to_opt[29]
        to_opt[29] = solution.reshape(to_opt_1.shape)


        self.model.set_weights(to_opt)

        ypred = np.argmax(self.model.predict(self.x_test), axis=1)

        # Evaluate the model's performance and return a metric value (Pre)
        A = Evaluation_Metrics(self.y_test, ypred)

        return A[0]

    def main_weight_updation_optimization(self, curr_wei):
        """
        Perform weight optimization using various optimization algorithms.

        Args:
        - curr_wei: Current model weights to be optimized.
        - opt: An integer representing the optimization algorithm choice.

        Returns:
        - best_position2: The optimized weights.
        """
        # Define problem parameters for optimization
        problem_dict = {
            "bounds": FloatVar(lb=sum((curr_wei - 0.01).tolist(), []), ub=sum((curr_wei + 0.01).tolist(), []),
                               name="delta"),
            "obj_func": self.fitness_function1,
            "minmax": "max",
        }

        # Use Hybrid Optimization algorithm
        cprint("[🚀🚀] Hybrid Optimization ", 'magenta', on_color='on_grey')
        model = HYBRID(epoch=10, pop_size=50)
        best_position = model.solve(problem_dict)
        return best_position.solution

    def main_update_hyperparameters(self):
        """
        Update model hyperparameters based on optimization results.

        Args:
        - opt: An integer representing the optimization algorithm choice.

        Returns:
        - self.model: The updated model with optimized weights.
        """
        # Get the current model weights
        to_opt = self.model.get_weights()

        to_opt_1 = to_opt[29]
        to_opt_2 = to_opt_1

        # Perform weight optimization and get the optimized weights
        wei_to_train_1 = self.main_weight_updation_optimization(to_opt_2)

        # Update the model weights with the optimized weights
        to_opt[29] = wei_to_train_1.reshape(to_opt_1.shape)
        self.model.set_weights(to_opt)

        # Return the updated model
        return self.model
