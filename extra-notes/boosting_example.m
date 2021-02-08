% BOOSTING_EXAMPLE
%
% Constructs a 2-dimensional dataset classifiable by boosting, but not any
% simple linear classifier, because of the thresholding nature of the data.

rand('seed', 0);

% m datapoints in 2-dimensions
mm = 150;
X = rand(mm, 2);

thresh_pos = .6;
y = [X(:, 1) < thresh_pos & X(:, 2) < thresh_pos];
y = 2 * y - 1;

for T = [2, 4, 5, 10]
  figure;
  hpos = plot(X(y == 1, 1), X(y == 1, 2), 'o');
  hold on;
  hneg = plot(X(y == -1, 1), X(y == -1, 2), 'x');
  set(hpos, 'linewidth', 2);
  set(hneg, 'linewidth', 2);

  [theta, feature_inds, thresholds] = stump_booster(X, y, T);

  x1_coords = linspace(0, 1, 100);
  x2_coords = linspace(0, 1, 100);
  Z = zeros(100);
  for ii = 1:100
    for jj = 1:100
      pred = (sign(x1_coords(ii) - thresholds(feature_inds == 1))' * ...
              theta(feature_inds == 1)) + ...
             (sign(x2_coords(jj) - thresholds(feature_inds == 2))' * ...
              theta(feature_inds == 2));
      Z(jj, ii) = sign(pred);
    end
  end

  C = contourc(x1_coords, x2_coords, Z, [0 0]);
  h = plot(C(1, 2:end), C(2, 2:end), 'k-');
  set(h, 'linewidth', 2);
  title(sprintf('Iterations = %d', T));
  set(gca, 'fontsize', 18);
  print('-depsc2', sprintf('boost_plot_%d.eps', T));
end

%% Now solve the logistic regression problem directly

mm = 200;
X = rand(mm, 2);
y = [X(:, 1) < thresh_pos & X(:, 2) < thresh_pos];
y = 2 * y - 1;

theta_log = zeros(3, 1);
X_logit = [ones(mm, 1), X];
for iter = 1:1000
  risk = (1/mm) * sum(log(1 + exp(-y .* (X_logit * theta_log))));
  if (mod(iter, 50) == 0)
    fprintf(1, 'Iter %d, loss %1.4f\n', iter, risk);
  end
  p = 1 ./ (1 + exp(y .* (X_logit * theta_log)));
  g = -(1/mm) * X_logit' * (p .* y);
  theta_log = theta_log - 2 * g;
end

x1_coord = linspace(0, 1, 100);
x2_coord = -(theta_log(1) + theta_log(2) * x1_coord) / theta_log(3);

figure;
hpos = plot(X(y == 1, 1), X(y == 1, 2), 'o');
hold on;
hneg = plot(X(y == -1, 1), X(y == -1, 2), 'x');
set(hpos, 'linewidth', 2);
set(hneg, 'linewidth', 2);
h = plot(x1_coord, x2_coord, 'k-', 'linewidth', 2);
axis([0 1 0 1]);
set(gca, 'fontsize', 18);
print -depsc2 'logistic_plot.eps';
