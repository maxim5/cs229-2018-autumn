# Important note: you do not have to modify this file for your homework.

# import util
from PS2.src import util
import numpy as np

MAX_EPOCHS = 400000
def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 100
    grads = []

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            grads.append([i, grad[1], grad[2]])
            # print('Finished %d iterations' % i)
        if i == MAX_EPOCHS:
            print(f"Could not converge in {MAX_EPOCHS} epochs")
            break
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return np.array(grads)


def calc_loss(X, Y, theta, l2_reg_coef):
    m, n = X.shape
    xytheta = Y * X.dot(theta)
    l2_norm = (1/m) * l2_reg_coef * np.sum(theta**2)
    loss = (1/m)*np.sum(np.log(np.exp(xytheta) + 1) - xytheta) + l2_norm
    return loss

def logistic_regression_modified(X, Y, loss_func = calc_loss, grad_func = calc_grad, 
    l2_reg_coef=0, max_iters=10000, log_step=10000, learning_rate=10, decay=lambda i, lr: lr):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    grads = []
    thetas = []
    norms = []
    losses = []

    i = 0
    while True:
        i += 1
        prev_theta = theta
        learning_rate = decay(i, learning_rate)
        grad = grad_func(X, Y, theta)
        grad[1:] = grad[1:] + (1/m)*l2_reg_coef * theta[1:]
        theta = theta - learning_rate * grad
        norm = np.linalg.norm(prev_theta - theta)
        loss = loss_func(X, Y, theta, l2_reg_coef)
        
        if i % log_step == 0:
            grads.append([i, grad[1], grad[2]])
            thetas.append([i, theta[1], theta[2]])
            norms.append([i, norm])
            losses.append([i, loss])
            # print(f"iterations: {i}, norm: {np.linalg.norm(prev_theta - theta)}")
            # print('Finished %d iterations' % i)
        
        if i == max_iters:
            print(f"Could not converge in {max_iters} epochs")
            break
        if norm < 1e-15:
            print('Converged in %d iterations' % i)
            grads.append([i, grad[1], grad[2]])
            thetas.append([i, theta[1], theta[2]])
            norms.append([i, norm])
            losses.append([i, loss])
            break
    
    return np.array(grads), np.array(thetas), np.array(norms), np.array(losses)


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()