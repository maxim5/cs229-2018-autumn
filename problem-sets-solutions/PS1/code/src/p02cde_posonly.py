import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    #######################################################################################
    # Problem (c)
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model_t = LogisticRegression()
    model_t.fit(x_train, t_train)

    util.plot(x_test, t_test, model_t.theta, 'output/p02c.png')

    t_pred_c = model_t.predict(x_test)
    np.savetxt(pred_path_c, t_pred_c > 0.5, fmt='%d')
    #######################################################################################
    # Problem (d)
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    model_y = LogisticRegression()
    model_y.fit(x_train, y_train)

    util.plot(x_test, y_test, model_y.theta, 'output/p02d.png')

    y_pred = model_y.predict(x_test)
    np.savetxt(pred_path_d, y_pred > 0.5, fmt='%d')
    #######################################################################################  
    # Problem (e)
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    alpha = np.mean(model_y.predict(x_valid))

    correction = 1 + np.log(2 / alpha - 1) / model_y.theta[0]
    util.plot(x_test, t_test, model_y.theta, 'output/p02e.png', correction)

    t_pred_e = y_pred / alpha
    np.savetxt(pred_path_e, t_pred_e > 0.5, fmt='%d')
    #######################################################################################
    # *** END CODER HERE
