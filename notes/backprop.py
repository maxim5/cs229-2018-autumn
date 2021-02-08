import numpy as np
from copy import copy

# Example backpropagation code for binary classification with 2-layer
# neural network (single hidden layer)

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def fprop(x, y, params):
  # Follows procedure given in notes
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  z1 = np.dot(W1, x) + b1
  a1 = sigmoid(z1)
  z2 = np.dot(W2, a1) + b2
  a2 = sigmoid(z2)
  loss = -(y * np.log(a2) + (1-y) * np.log(1-a2))
  ret = {'x': x, 'y': y, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'loss': loss}
  for key in params:
    ret[key] = params[key]
  return ret

def bprop(fprop_cache):
  # Follows procedure given in notes
  x, y, z1, a1, z2, a2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'a1', 'z2', 'a2', 'loss')]
  dz2 = (a2 - y)
  dW2 = np.dot(dz2, a1.T)
  db2 = dz2
  dz1 = np.dot(fprop_cache['W2'].T, dz2) * sigmoid(z1) * (1-sigmoid(z1))
  dW1 = np.dot(dz1, x.T)
  db1 = dz1
  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

# Gradient checking

if __name__ == '__main__':
  # Initialize random parameters and inputs
  W1 = np.random.rand(2,2)
  b1 = np.random.rand(2, 1)
  W2 = np.random.rand(1, 2)
  b2 = np.random.rand(1, 1)
  params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
  x = np.random.rand(2, 1)
  y = np.random.randint(0, 2)  # Returns 0/1

  fprop_cache = fprop(x, y, params)
  bprop_cache = bprop(fprop_cache)

  # Numerical gradient checking
  # Note how slow this is! Thus we want to use the backpropagation algorithm instead.
  eps = 1e-6
  ng_cache = {}
  # For every single parameter (W, b)
  for key in params:
    param = params[key]
    # This will be our numerical gradient
    ng = np.zeros(param.shape)
    for j in range(ng.shape[0]):
      for k in xrange(ng.shape[1]):
        # For every element of parameter matrix, compute gradient of loss wrt
        # that element numerically using finite differences
        add_eps = np.copy(param)
        min_eps = np.copy(param)
        add_eps[j, k] += eps
        min_eps[j, k] -= eps
        add_params = copy(params)
        min_params = copy(params)
        add_params[key] = add_eps
        min_params[key] = min_eps
        ng[j, k] = (fprop(x, y, add_params)['loss'] - fprop(x, y, min_params)['loss']) / (2 * eps)
    ng_cache[key] = ng

  # Compare numerical gradients to those computed using backpropagation algorithm
  for key in params:
    print key
    # These should be the same
    print(bprop_cache[key])
    print(ng_cache[key])
