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
    x_train, t_train = util.load_dataset(train_path, label_col="t", add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col="t", add_intercept=True)

    x_train, y_train = util.load_dataset(train_path, label_col="y", add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col="y", add_intercept=True)

    x_valid, y_valid = util.load_dataset(train_path, label_col="y", add_intercept=True)

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    logreg_model = LogisticRegression()
    theta = logreg_model.fit(x_train, t_train)
    t_fcsts_c, _ = logreg_model.predict(x_test)
    util.evaluate_classification(t_fcsts_c, t_test, "PS1 p02 (c)")
    util.write(t_fcsts_c, pred_path_c)

    fig_path_prefix = pred_path_c.split(".")[0]
    util.plot(x_test, t_test, theta, save_path=f"{fig_path_prefix}_fig")

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    logreg_model = LogisticRegression()
    theta = logreg_model.fit(x_train, y_train)
    t_fcsts_d, _ = logreg_model.predict(x_test)
    util.evaluate_classification(t_fcsts_d, t_test, "PS1 p02 (d)")
    util.write(t_fcsts_d, pred_path_d)

    fig_path_prefix = pred_path_d.split(".")[0]
    util.plot(x_test, t_test, theta, save_path=f"{fig_path_prefix}_fig")

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    pos_examples = np.asarray(y_valid, dtype="int")
    v_pos_x = x_valid[pos_examples]
    _, raw_probs = logreg_model.predict(x_valid)
    a = np.sum(raw_probs)/len(raw_probs)
    print(f"\n\nalpha = {a}")
    _, raw_probs = logreg_model.predict(x_test)
    adjusted_probs = raw_probs/a
    fcsts = adjusted_probs >= 0.5
    util.evaluate_classification(fcsts, t_test, "PS1 p02 (e)")
    util.write(fcsts, pred_path_e)

    fig_path_prefix = pred_path_e.split(".")[0]
    util.plot(x_test, t_test, theta, save_path=f"{fig_path_prefix}_fig",
              correction=-np.log((2/a) - 1)/theta[2])
    # util.plot(x_test, t_test, theta, save_path=f"{fig_path_prefix}_fig",
    #           correction=a)
    # *** END CODER HERE
