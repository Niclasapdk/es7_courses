%% Problem 2: Binomial and Poisson Point Processes
clc; clear; close all;

%% a) Simulate and plot realizations 
% --- Define region ---
Sx = [0 2];
Sy = [0 2];
n = 1e3;  % number of points

% --- Define PDF function ---
% For now: uniform over S (1/area)
f = @(x,y) (x >= 0 & x <= 2 & y >= 0 & y <= 2) * (1/4);

% If you want a non-uniform example later, try this:
% f = @(x,y) (x >= 0 & x <= 2 & y >= 0 & y <= 2) .* (x + y) / 8;

% --- Generate random samples according to f(x,y) ---
Ntrial = 1e5;  % number of trial points for rejection sampling
x_trial = 2 * rand(Ntrial,1);
y_trial = 2 * rand(Ntrial,1);

% Compute f(x,y) for trial points
f_vals = arrayfun(f, x_trial, y_trial);
f_max = max(f_vals);  % upper bound for rejection sampling

% Rejection sampling
accepted_x = [];
accepted_y = [];

while numel(accepted_x) < n
    x_cand = 2 * rand;
    y_cand = 2 * rand;
    u = rand * f_max;   % random height
    if u <= f(x_cand, y_cand)
        accepted_x(end+1,1) = x_cand;
        accepted_y(end+1,1) = y_cand;
    end
end

% --- Plot ---
figure;
hold on; grid on; axis equal;
rectangle('Position',[0 0 2 2],'EdgeColor','k','LineWidth',1.5);
plot(accepted_x, accepted_y, 'bo', 'MarkerFaceColor', 'b');
xlabel('x_1'); ylabel('x_2');
title('Binomial Point Process (12 samples) from PDF f(x_1,x_2)');
% it is uniform disb
% The S region is 2x2 = 4
% Where Density is 1/4
% Means uniformly distributed points over the square

%% (b) Region count distribution for B1 = [0,1]x[0,1]
n = 12;              % total number of points in the BPP
p = 1/4;             % probability that a point falls in B1
x = 0:n;             % possible counts

% Binomial PMF
pmf = binopdf(x, n, p);

% Plot PMF
figure;
stem(x, pmf, 'b', 'LineWidth', 1.5);
xlabel('k = number of points in B1');
ylabel('P(N_X(B1) = k)');
title('PMF of N_X(B1) ~ Binomial(12, 1/4)');
grid on;

% Expected value and variance
mean_NB1 = n * p;
var_NB1 = n * p * (1 - p);

fprintf('Exercise 14.2 b)\n')
fprintf('Expected number of points in B1 = %.2f\n', mean_NB1);
fprintf('Variance = %.2f\n', var_NB1);

%% (c) Empirical distribution of N_X(B1) over many realizations

% Simulation parameters
numRealizations = 1e4;  % number of independent realizations
n = 12;                 % number of points in each realization
Sx = [0 2];
Sy = [0 2];
B1x = [0 1];
B1y = [0 1];

NX_B1 = zeros(numRealizations,1);  % store counts

for i = 1:numRealizations
    % Generate one realization (n uniform points in S)
    x = 2 * rand(n,1);
    y = 2 * rand(n,1);

    % Count how many points fall inside B1
    inside = (x >= B1x(1) & x <= B1x(2)) & (y >= B1y(1) & y <= B1y(2));
    NX_B1(i) = sum(inside);
end

% --- Plot histogram ---
figure;
histogram(NX_B1, 'Normalization', 'pdf', 'FaceColor', [0.3 0.7 0.9]);
hold on;

% --- Theoretical Binomial PMF ---
xvals = 0:n;
p = 1/4;
pmf = binopdf(xvals, n, p);
stem(xvals, pmf, 'r', 'LineWidth', 1.5);
legend('Empirical (histogram)', 'Theoretical Binomial PMF', 'Location', 'best');
xlabel('k = number of points in B1');
ylabel('Probability');
title('Empirical Distribution of N_X(B1) vs. Theoretical Binomial PMF');
grid on;

% --- Print mean and variance comparison ---
mean_emp = mean(NX_B1);
var_emp = var(NX_B1);

fprintf('\nExercise 14.2 (c)\n');
fprintf('Empirical mean of N_X(B1) = %.2f (theoretical = %.2f)\n', mean_emp, n*p);
fprintf('Empirical variance of N_X(B1) = %.2f (theoretical = %.2f)\n', var_emp, n*p*(1-p));

%% (d) Joint region counts for B1 and B2 = S \ B1

% Simulation parameters
numRealizations = 1e4;
n = 12;
Sx = [0 2];
Sy = [0 2];
B1x = [0 1];
B1y = [0 1];

NX_B1 = zeros(numRealizations,1);
NX_B2 = zeros(numRealizations,1);

for i = 1:numRealizations
    % Generate one realization (n uniform points in S)
    x = 2 * rand(n,1);
    y = 2 * rand(n,1);

    % Count points in B1
    inside_B1 = (x >= B1x(1) & x <= B1x(2)) & (y >= B1y(1) & y <= B1y(2));
    NX_B1(i) = sum(inside_B1);

    % Points in B2 = S \ B1 (the rest)
    NX_B2(i) = n - NX_B1(i);
end

% --- Scatter plot ---
figure;
scatter(NX_B1, NX_B2, 10, 'filled');
xlabel('N_X(B_1)');
ylabel('N_X(B_2)');
title('Scatter of Region Counts: N_X(B1) vs N_X(B2)');
grid on;

% --- Compute correlation ---
corr_val = corr(NX_B1, NX_B2);

fprintf('\nExercise 14.2 (d)\n');
fprintf('Correlation between N_X(B1) and N_X(B2): %.3f\n', corr_val);

% The scatter plot will form a perfect diagonal line with negative slope
% (because they are fully negatively correlated)
%% Visualize one realization showing S, B1, and B2 regions

n = 12; % number of points per realization
Sx = [0 2];
Sy = [0 2];
B1x = [0 1];
B1y = [0 1];

% --- Generate one realization ---
x = 2 * rand(n,1);
y = 2 * rand(n,1);

% --- Classify points ---
inside_B1 = (x >= B1x(1) & x <= B1x(2)) & (y >= B1y(1) & y <= B1y(2));
inside_B2 = ~inside_B1;

% --- Plot ---
figure;
hold on; axis equal; grid on;

% Region S
rectangle('Position',[Sx(1) Sy(1) diff(Sx) diff(Sy)], ...
          'EdgeColor','k','LineWidth',1.5);

% Subregion B1
rectangle('Position',[B1x(1) B1y(1) diff(B1x) diff(B1y)], ...
          'EdgeColor','b','LineWidth',2);

% Plot points in B1 and B2 with different colors
plot(x(inside_B1), y(inside_B1), 'bo', 'MarkerFaceColor', 'b', 'DisplayName','Points in B1');
plot(x(inside_B2), y(inside_B2), 'ro', 'MarkerFaceColor', 'r', 'DisplayName','Points in B2');

xlabel('x_1'); ylabel('x_2');
title('One Realization of Binomial Point Process: B1 and B2 Highlighted');
legend show;

% --- Count results ---
NX_B1 = sum(inside_B1);
NX_B2 = sum(inside_B2);

fprintf('\nExercise 14.2 (e)\n');
fprintf('Points in B1 = %d, Points in B2 = %d (Total = %d)\n', NX_B1, NX_B2, n);




%% (e) Generate and plot realizations of a Poisson Point Process (PPP)

clc;
% Area of S = 4
lambda0 = 3;  % intensity (points per unit area)
% Mean for poission = Area(s) * lambda0
Sx = [0 2];
Sy = [0 2];
areaS = diff(Sx) * diff(Sy);

% Random number of points ~ Poisson(lambda0 * area)
N_points = poissrnd(lambda0 * areaS);

% Generate uniform point locations
xY = 2 * rand(N_points,1);
yY = 2 * rand(N_points,1);

% Plot realization
figure;
hold on; axis equal; grid on;
rectangle('Position',[0 0 2 2],'EdgeColor','k','LineWidth',1.5);
plot(xY, yY, 'ko', 'MarkerFaceColor','g');
xlabel('y_1'); ylabel('y_2');
title(sprintf('Poisson Point Process: λ₀ = %.1f, N = %d points', lambda0, N_points));

fprintf('\nExercise (e)-(f)\n');
fprintf('Mean expected points = λ₀ × Area = %.2f × 4 = %.2f\n', lambda0, lambda0*areaS);
fprintf('Actual points in this realization: %d\n', N_points);

% Poisson random variables are random — they fluctuate around the mean.
%% (g) Region count distribution for B1 = [0,1]x[0,1]
B1x = [0 1];
B1y = [0 1];
areaB1 = diff(B1x) * diff(B1y);

% The region count N_Y(B1) ~ Poisson(λ₀ × |B1|)
lambda_B1 = lambda0 * areaB1;
xvals = 0:15;
pmf = poisspdf(xvals, lambda_B1);

% Plot PMF
figure;
stem(xvals, pmf, 'r', 'LineWidth', 1.5);
xlabel('k = number of points in B1');
ylabel('P(N_Y(B1) = k)');
title(sprintf('PMF of N_Y(B1) ~ Poisson(λ₀ × |B1|) = Poisson(%.2f)', lambda_B1));
grid on;

fprintf('Exercise (g)\n');
fprintf('Expected number of points in B1 = %.2f\n', lambda_B1);
fprintf('Variance = %.2f\n', lambda_B1);

%% (h) Empirical histogram of region counts over many realizations

numRealizations = 1e4;
NY_B1 = zeros(numRealizations,1);

for i = 1:numRealizations
    % Random number of points in this realization
    N_pts = poissrnd(lambda0 * areaS);
    x = 2 * rand(N_pts,1);
    y = 2 * rand(N_pts,1);

    % Count how many fall inside B1
    inside_B1 = (x >= B1x(1) & x <= B1x(2)) & (y >= B1y(1) & y <= B1y(2));
    NY_B1(i) = sum(inside_B1);
end

% --- Histogram vs PMF ---
figure;
histogram(NY_B1, 'Normalization', 'pdf', 'FaceColor', [0.4 0.8 0.4]);
hold on;
stem(xvals, pmf, 'r', 'LineWidth', 1.5);
xlabel('k = number of points in B1');
ylabel('Probability');
title('Empirical Histogram of N_Y(B1) vs Theoretical Poisson PMF');
legend('Empirical', 'Theoretical PMF');
grid on;

fprintf('Exercise (h)\n');
fprintf('Empirical mean = %.2f, theoretical mean = %.2f\n', mean(NY_B1), lambda_B1);
fprintf('Empirical variance = %.2f, theoretical variance = %.2f\n', var(NY_B1), lambda_B1);

%% (i) Scatter plot of region counts N_Y(B1) vs N_Y(B2)
B2_area = areaS - areaB1;  % = 3
NY_B2 = zeros(numRealizations,1);

for i = 1:numRealizations
    N_pts = poissrnd(lambda0 * areaS);
    x = 2 * rand(N_pts,1);
    y = 2 * rand(N_pts,1);

    inside_B1 = (x >= B1x(1) & x <= B1x(2)) & (y >= B1y(1) & y <= B1y(2));
    NY_B1(i) = sum(inside_B1);
    NY_B2(i) = N_pts - NY_B1(i);
end

figure;
scatter(NY_B1, NY_B2, 10, 'filled');
xlabel('N_Y(B_1)');
ylabel('N_Y(B_2)');
title('Scatter of Region Counts for Poisson Point Process');
grid on;

% Correlation
corr_val = corr(NY_B1, NY_B2);

fprintf('Exercise (i)\n');
fprintf('Correlation between N_Y(B1) and N_Y(B2): %.3f\n', corr_val);


% This is exactly what we expect from a PoissonPP, because 
% N_Y(B1) and N_Y(B2) are independent Poisson random variables.
%
% Most of the points are clustered around:
%
% N_Y(B1) ≈ 3   (mean for B1, area = 1 × 1, λ₀ = 3)
% N_Y(B2) ≈ 9   (mean for B2, area = 3, λ₀ = 3)
%
% There's no strict negative correlation, unlike the BinomialPP case.
% Because the PoissonPP has a random (potentially infinite) number of points,
% counts in different regions are independent, so knowing N_Y(B1) gives no 
% information about N_Y(B2).
