import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data

    fig_path_prefix = util.get_fig_prefix(pred_path)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    mse_vals = []

    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)

        # Get MSE value on the validation set
        fcsts = model.predict(x_valid)
        mse = util.evaluate_regression(y_valid, fcsts,
                                   f"PS1 p05(c) tau: {tau} valid_set")
        tau_str = str(tau).split(".")
        tau_str = "_".join(tau_str)
        plot_path = f"{fig_path_prefix}_tau_{tau_str}"
        util.plot_regression_train_and_fcst(x_valid, y_valid, x_valid, fcsts,
                                            save_path=plot_path)
        mse_vals.append(mse)

    # Plot validation predictions on top of test set
    best_tau = tau_values[np.argmin(mse_vals)]
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)

    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    fcsts = model.predict(x_test)
    mse = util.evaluate_regression(y_test, fcsts,
                             f"PS1 p05(c) tau: {best_tau}, test_set")
    # *** END CODE HERE ***
