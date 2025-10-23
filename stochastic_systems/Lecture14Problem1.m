%% Problem 1: Binomial and Poisson Random Variables
clc; clear; close all;

%% (a) Binomial Distribution
N = 10;      % number of trials -- The amount of times we throw the dice in 1 realization
p = 0.3;     % success probability
numSamples = 1e5;  % number of i.i.d. samples   --- How many times we generate the bi nomial (how many times we throw the dice N times) Amount of realization

% Generate random samples
binomial_samples = binornd(N, p, [numSamples, 1]);

% Plot histogram (normalized)
figure;
histogram(binomial_samples, 'Normalization', 'pdf', 'FaceColor', [0.3 0.7 0.9]);
hold on;

% Theoretical PMF
x = 0:N;
pmf_binomial = binopdf(x, N, p);
stem(x, pmf_binomial, 'r', 'LineWidth', 1.5);
title(sprintf('Binomial Distribution: N=%d, p=%.2f', N, p));
xlabel('k'); ylabel('Probability');
legend('Empirical (histogram)', 'Theoretical PMF');
grid on;

%% (b) Poisson Distribution
mu = 3;      % mean
numSamples = 1e5;

% Generate random samples
poisson_samples = poissrnd(mu, [numSamples, 1]);

% Plot histogram (normalized)
figure;
histogram(poisson_samples, 'Normalization', 'pdf', 'FaceColor', [0.3 0.9 0.5]);
hold on;

% Theoretical PMF
x = 0:max(poisson_samples);
pmf_poisson = poisspdf(x, mu);
stem(x, pmf_poisson, 'r', 'LineWidth', 1.5);
title(sprintf('Poisson Distribution: \\mu = %.1f', mu));
xlabel('k'); ylabel('Probability');
legend('Empirical (histogram)', 'Theoretical PMF');
grid on;

%% (c) Binomial vs Poisson Convergence
mu = 3;    % fixed mean for comparison
N_values = [10, 30, 100, 500];  % increasing N
figure;
hold on;

for i = 1:length(N_values)
    N = N_values(i);
    p = mu / N;             % success probability p = µ / N
    x = 0:15;               % range of k values
    pmf_binomial = binopdf(x, N, p);
    plot(x, pmf_binomial, 'DisplayName', sprintf('N = %d', N));
end

% Poisson reference
pmf_poisson = poisspdf(x, mu);
plot(x, pmf_poisson, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Poisson (\mu=3)');

title('Binomial PMF convergence to Poisson PMF');
xlabel('k'); ylabel('Probability');
legend show;
grid on;

%The Poisson distribution can actually be seen as a limit case of the Binomial distribution.

%When the number of opportunities (N) becomes huge,
%and each opportunity’s chance of success (p) is tiny —
%but overall average count (μ = Np) stays the same —
%then the Binomial behaves exactly like a Poisson.
