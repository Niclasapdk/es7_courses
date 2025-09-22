% Assignment 2: NLMS.
% Author: Iván López-Espejo (ivl@es.aau.dk).

clear;
close all
clc

% Loading signals...
load local.asc
load remota.asc
remota = remota';
load signal.asc

% 1. SNR with no echo cancellation.
SNR = 10*log10(sum(local.^2) / sum((local - signal).^2));
disp(SNR)

% 2a. Looking for the best NLMS configuration.
mus = 1e-4 * (2.^(0:9));
ps = 1:10;
SNRs = zeros(length(mus), length(ps));
for i = 1:length(mus)
    for j = 1:length(ps)
        
        % Parameters.
        mu = mus(i);
        p = ps(j);
        
        % Initialization.
        w = zeros(p,1);
        x = zeros(p,1);  % Signal fragment to be filtered.
        e = zeros(size(remota));
        
        % Filtering.
        for k = 1:length(remota)
            x(2:end) = x(1:end-1);
            x(1) = remota(k);
            sig = sum(x.^2);
            e(k) = signal(k) - w' * x;
            w = w + (mu / (sig + 1e-10)) * x * e(k);
        end
        
        % SNR computation.
        SNRs(i,j) = 10*log10(sum(local.^2) / sum((local - e).^2));
        
    end
end

% 2b. We draw the time evolution of the filter weights for the best SNR
% configuration.
maximum = max(max(SNRs));
[posmu, posp] = find(SNRs==maximum);
opt_mu = mus(posmu);  % Optimal step-size parameter.
opt_p = ps(posp);  % Optimal filter order.
% Initialization.
w = zeros(opt_p,length(remota)+1);
x = zeros(opt_p,1);  % Signal fragment to be filtered.
e = zeros(size(remota));
% Filtering.
for k = 1:length(remota)
    x(2:end) = x(1:end-1);
    x(1) = remota(k);
    sig = sum(x.^2);
    e(k) = signal(k) - w(:,k)' * x;
    w(:,k+1) = w(:,k) + (opt_mu / (sig + 1e-10)) * x * e(k);
end
figure
plot(w')
grid on
title('Weights')
ylabel('Magnitude')
xlabel('Cycle (n)')
legend('w_0(n)','w_1(n)','w_2(n)','w_3(n)','w_4(n)')
disp(maximum)
figure
plot(signal)
hold on
plot(e,'r--')
hold off
legend('Received','Filtered')
grid on
ylabel('Amplitude')
xlabel('Time (n)')

% 3a. Looking for the best NLMS configuration.
mus = 1e-4 * (2.^(0:9));
ps = 1:10;
SNRs = zeros(length(mus), length(ps));
for i = 1:length(mus)
    for j = 1:length(ps)
        
        % Parameters.
        mu = mus(i);
        p = ps(j);
        
        % Initialization.
        w = zeros(p,1);
        x = zeros(p,1);  % Signal fragment to be filtered.
        e = zeros(size(remota));
        
        % Filtering.
        for k = 1:length(remota)
            x(2:end) = x(1:end-1);
            x(1) = remota(k);
            sig = sum(x.^2);
            e(k) = signal(k) - w' * x;
            if k <= 2200
                w = w + (mu / (sig + 1e-10)) * x * e(k);
            end
        end
        
        % SNR computation.
        SNRs(i,j) = 10*log10(sum(local.^2) / sum((local - e).^2));
        
    end
end

% 3b. We draw the time evolution of the filter weights for the best SNR
% configuration.
maximum = max(max(SNRs));
[posmu, posp] = find(SNRs==maximum);
opt_mu = mus(posmu);  % Optimal step-size parameter.
opt_p = ps(posp);  % Optimal filter order.
% Initialization.
w = zeros(opt_p,length(remota)+1);
x = zeros(opt_p,1);  % Signal fragment to be filtered.
e = zeros(size(remota));
% Filtering.
for k = 1:length(remota)
    x(2:end) = x(1:end-1);
    x(1) = remota(k);
    sig = sum(x.^2);
    e(k) = signal(k) - w(:,k)' * x;
    if k <= 2200
        w(:,k+1) = w(:,k) + (opt_mu / (sig + 1e-10)) * x * e(k);
    else
        w(:,k+1) = w(:,k);
    end
end
figure
plot(w')
grid on
title('Weights (DTD)')
ylabel('Magnitude')
xlabel('Cycle (n)')
legend('w_0(n)','w_1(n)','w_2(n)','w_3(n)','w_4(n)')
disp(maximum)
figure
plot(signal)
hold on
plot(e,'r--')
hold off
legend('Received','Filtered')
grid on
ylabel('Amplitude')
xlabel('Time (n)')

% Frequency response of the estimated filter.
wf = w(:, end);
fr = 20*log10(abs(freqz(wf,1,257)));
x_range = 0:4000/256:4000;
figure
plot(x_range,fr)
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')
title('Filter response (DTD)')
grid on