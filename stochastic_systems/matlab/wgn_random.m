% Parameters
N = 50; % Number of samples
M = 50; % Number of realizations
sigma = 1; % Standard deviation

% Generate and plot realizations
figure;
hold on;
for m = 1:M
    X = sigma * randn(1, N); % Generate WGN realization
    n = 0:N-1; % Time index
    plot(n, X, '.', 'MarkerSize', 10); % Scatter plot
end
hold off;

% Labels and title
xlabel('n');
ylabel('X[n]');
title('50 Realizations of WGN Random Process');
grid on;