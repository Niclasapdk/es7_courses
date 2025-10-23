% This script is for Stochastic processes Lecture 14,
% 
% Mathias Jørgensen, AAU, Oct. 2025

% Simulation Setup
% Intensity function. Does not change the pulses,
% but changes the number of pulses.
% So large rho gives more pulses
rho0 = 1; 
tMin = 0;
tMax = 10;

nMonteCarlo = 10000;
nTimeSample = 1000;
time = linspace(tMin,tMax,nTimeSample);
Zsample = zeros(nMonteCarlo,nTimeSample);

% Pulse function
h =  @(t) (t>=0).*(t<=1).*(-4.*t.^2+4.*t); % Anonymous function 

% Run Monte Carlo simulation
% The Monte Carlo simulation generates a number of random realizations,
% where one realization is the sum of pulse functions, at random time intervals
% The number of pulses in the realization is generated from poisson.
for iMonteCarlo = 1:nMonteCarlo
% Draw homogeneous poisson Process
N  = random('Poisson',(tMax-tMin)*rho0); 
tau = rand(N,1)*tMax;

% Make annonymoys function for shot noise:
Z = @(t) 0; 
for index = 1:N
Z = @(t) Z(t) + h(t-tau(index));
end
% Evaluate and store function values into matrix
Zsample(iMonteCarlo,:) = Z(time);
end

% Plotting
% We see a single deterministic pulse h(t):
subplot(411)
plot(time,h(time))
xlabel('t [s]')
ylabel('h(t)')
title('The pulse h(t) ')

% One Monte Carlo realization, that sums the pulse functions
subplot(412)
plot(time,Z(time))
xlabel('t [s]')
ylabel('Z(t)')
title('One realization')

% For a stationary process we expect the mean of Z(t) to be constant over time, here it equals 2/3.
% 2x rho gives 2x mean.
subplot(413)
plot(time,mean(Zsample))
xlabel('t [s]')
ylabel('mean(Z(t))')
title('Estimated mean of Z(t)')

% Shows all the values for t=10(t_max) of Z(t). The realizations are mostly
% 0. The probability is cumulatative.
% A higher rho, gives larger values, therefore, the norm plot has higher
% data values, and less prominent probability of 0.
subplot(414)
normplot(Zsample(:,end))

% Exercise d

% For each time t, integrate h(tau) over shifted interval
E_Zt = zeros(size(time));
for k = 1:length(time)
    t = time(k);
    tauLimits = [tMin, tMax];
    % Integrate h(t - tau) over tau from tMin to tMax
    E_Zt(k) = rho0 * integral(@(tau) h(t - tau), tauLimits(1), tauLimits(2));
end

% Plot
figure;
plot(time, mean(Zsample), 'b', 'LineWidth', 1.5)
hold on
plot(time, E_Zt, 'r--', 'LineWidth', 1.5)
xlabel('t [s]')
ylabel('Z(t)')
legend('Estimated mean', 'Theoretical mean')
title('Estimated vs Theoretical Mean of Z(t)')
grid on
