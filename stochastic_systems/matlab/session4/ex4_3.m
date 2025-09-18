% Parameters
num_samples = 1000; % Number of samples for simulation
num_realizations = 100; % Number of realizations to simulate
t = linspace(0, 2*pi, num_samples); % Time vector

% Preallocate arrays for mean and variance
mean_X = zeros(size(t));
var_X = zeros(size(t));

% Monte Carlo simulation 
for i = 1:num_realizations
    Theta = unifrnd(-pi, pi); % Sample Theta from uniform distribution
    X_t = cos(t + Theta); % Generate realization of X(t)
    
    % Accumulate mean and variance
    mean_X = mean_X + X_t;
    var_X = var_X + X_t.^2;
    
    % Plot each realization
    plot(t, X_t);
    hold on;
end

% Finalize the plot
hold off;
xlabel('Time (t)');
ylabel('X(t)');
title('Realizations of X(t) = cos(t + Theta)');
grid on;

% Compute the mean and variance
mean_X = mean_X / num_realizations; % Mean function
var_X = (var_X / num_realizations) - (mean_X.^2); % Variance function

% Display mean and variance
figure;
subplot(2, 1, 1);
plot(t, mean_X);
title('Mean Function of X(t)');
xlabel('Time (t)');
ylabel('Mean');

subplot(2, 1, 2);
plot(t, var_X);
title('Variance Function of X(t)');
xlabel('Time (t)');
ylabel('Variance');