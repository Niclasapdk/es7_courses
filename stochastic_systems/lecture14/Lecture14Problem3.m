%% Exercise a-c
% Inhomogeneous Poisson Point Process
clc; clear; close all;

% Define region
S = [0 1; 0 1]; % [x1min x1max; x2min x2max]

% Intensity function λ(x1, x2) = 30 * x1
lambda = @(x1, x2) 30 .* x1;

% Exercise a) 
% Expected number of points
expected_N = integral2(lambda, 0, 1, 0, 1);
fprintf('Expected number of points in S: %.2f\n', expected_N); % Expects 30*0.5=15 (x1 integrates from 0 to 1 giving 0.5)

% Exercise c)
% Simulate one realization
% Step 1: Draw total number of points
N = poissrnd(expected_N);

% Step 2: Generate x1, x2 according to λ(x)
x1 = sqrt(rand(N,1));   % Inverse CDF of f(x1) = 2x1
x2 = rand(N,1);         % Uniform in [0,1]

% Step 3: Plot realization
% Expects a plot that is dense at x1=1 and no points at x1=0, because
% lambda(x1) = 30*x1
% lambda(x1=0) = 0
% lambda(x1=1) = 30
figure;
scatter(x1, x2, 40, 'filled');
xlabel('x_1'); ylabel('x_2');
title(sprintf('Inhomogeneous Poisson PP (N = %d)', N));
axis([0 1 0 1]); grid on;

%% Exercise d)
% Modified intensity: λ̃(x) = 30 * x1 * 1[x1^2 + x2^2 <= 1]
clear; close all; clc;

lambda = @(x1, x2) 30 .* x1 .* (x1.^2 + x2.^2 <= 1);

% Expected number of points (numerical integration)
expected_N_mod = integral2(lambda, 0, 1, 0, 1);
fprintf('Expected number of points in modified region: %.2f\n', expected_N_mod);

% Simulate one realization
N = poissrnd(expected_N_mod);

% Sample candidates
x1 = sqrt(rand(N*2,1)); % oversample for rejection method
x2 = rand(N*2,1);

% Keep only those inside the unit circle
inside = (x1.^2 + x2.^2 <= 1);
x1 = x1(inside);
x2 = x2(inside);

% If we have more than needed, trim
if numel(x1) > N
    idx = randperm(numel(x1), N);
    x1 = x1(idx); x2 = x2(idx);
end

% Plot realization
% The realization still only depends on x1 as in last exercise.
% The new constraints make all the points to be inside a quarter circle, so
% indirectly dependency of x2
theta = linspace(0, pi/2, 100);
figure;
plot(cos(theta), sin(theta), 'k--'); hold on;
scatter(x1, x2, 40, 'filled');
xlabel('x_1'); ylabel('x_2');
title(sprintf('Modified Poisson PP (λ̃), N = %d', N));
axis([0 1 0 1]); axis equal; grid on;
