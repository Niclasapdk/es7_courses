function random_process_analysis()
    % Main analysis function with user input
    disp('=== Random Process Analysis ===');
    
    % Get user input
    N = input('Enter number of samples per realization: ');
    M = input('Enter number of realizations: ');
    a = input('Enter mean parameter a: ');
    b = input('Enter variance parameter b: ');
    lambda = input('Enter exponential rate parameter lambda: ');
    
    % Analyze with different numbers of realizations
    analyze_with_varying_realizations(N, a, b, lambda);
    
    % Detailed analysis with all realizations
    detailed_analysis(N, M, a, b, lambda);
end

function analyze_with_varying_realizations(N, a, b, lambda)
    % Analyze how estimation accuracy improves with more realizations
    realization_counts = [10, 50, 100, 500, 1000];
    colors = ['r', 'g', 'b', 'm', 'c'];
    
    figure('Position', [100, 100, 1200, 800]);
    
    for process_type = 1:3
        subplot(3, 2, (process_type-1)*2+1);
        hold on;
        subplot(3, 2, (process_type-1)*2+2);
        hold on;
    end
    
    for i = 1:length(realization_counts)
        M = realization_counts(i);
        color = colors(i);
        
        % Generate processes
        [X, Y, Z] = generate_processes(N, M, a, b, lambda);
        
        % Estimate mean and variance
        [mu_X_est, var_X_est] = estimate_moments(X);
        [mu_Y_est, var_Y_est] = estimate_moments(Y);
        [mu_Z_est, var_Z_est] = estimate_moments(Z);
        
        % Plot results
        plot_results(mu_X_est, var_X_est, mu_Y_est, var_Y_est, mu_Z_est, var_Z_est, ...
                     a, b, lambda, color, sprintf('M=%d', M));
    end
    
    % Add legends and titles
    for process_type = 1:3
        subplot(3, 2, (process_type-1)*2+1);
        legend('show');
        grid on;
        subplot(3, 2, (process_type-1)*2+2);
        legend('show');
        grid on;
    end
end

function detailed_analysis(N, M, a, b, lambda)
    % Detailed analysis with all realizations
    fprintf('\n=== Detailed Analysis with M = %d realizations ===\n', M);
    
    % Generate processes
    [X, Y, Z] = generate_processes(N, M, a, b, lambda);
    
    % Estimate mean and variance
    [mu_X_est, var_X_est] = estimate_moments(X);
    [mu_Y_est, var_Y_est] = estimate_moments(Y);
    [mu_Z_est, var_Z_est] = estimate_moments(Z);
    
    % Theoretical values
    [mu_X_theo, var_X_theo] = theoretical_moments('gaussian', a, b, lambda);
    [mu_Y_theo, var_Y_theo] = theoretical_moments('uniform', a, b, lambda);
    [mu_Z_theo, var_Z_theo] = theoretical_moments('exponential', a, b, lambda);
    
    % Display results
    display_results(mu_X_est, var_X_est, mu_X_theo, var_X_theo, 'Gaussian');
    display_results(mu_Y_est, var_Y_est, mu_Y_theo, var_Y_theo, 'Uniform');
    display_results(mu_Z_est, var_Z_est, mu_Z_theo, var_Z_theo, 'Exponential');
    
    % Plot final results
    plot_final_results(mu_X_est, var_X_est, mu_Y_est, var_Y_est, mu_Z_est, var_Z_est, ...
                       mu_X_theo, var_X_theo, mu_Y_theo, var_Y_theo, mu_Z_theo, var_Z_theo);
end

function [X, Y, Z] = generate_processes(N, M, a, b, lambda)
    % Generate M realizations of each process with N samples
    
    % Gaussian process: X ~ N(a, b)
    X = sqrt(b) * randn(M, N) + a;
    
    % Uniform process: Y ~ Uniform with mean a and variance b
    % For Uniform(c,d): mean = (c+d)/2 = a, variance = (d-c)^2/12 = b
    c = a - sqrt(3*b);
    d = a + sqrt(3*b);
    Y = c + (d - c) * rand(M, N);
    
    % Exponential process: Z ~ Exponential with mean a
    % Transformation from uniform
    uniform_samples = rand(M, N);
    Z = -log(1 - uniform_samples) / (1/a); % Mean = 1/lambda = a
end

function [mu_est, var_est] = estimate_moments(process)
    % Estimate mean and variance across realizations
    mu_est = mean(process, 1);  % Mean across rows (realizations)
    var_est = var(process, 0, 1); % Variance across rows
end

function [mu_theo, var_theo] = theoretical_moments(type, a, b, lambda)
    % Theoretical mean and variance
    switch type
        case 'gaussian'
            mu_theo = a * ones(1, 100);  % Constant mean
            var_theo = b * ones(1, 100); % Constant variance
            
        case 'uniform'
            mu_theo = a * ones(1, 100);
            var_theo = b * ones(1, 100);
            
        case 'exponential'
            mu_theo = a * ones(1, 100);  % Mean = 1/lambda
            var_theo = a^2 * ones(1, 100); % Variance = (1/lambda)^2
    end
end

function plot_results(mu_X_est, var_X_est, mu_Y_est, var_Y_est, mu_Z_est, var_Z_est, ...
                     a, b, lambda, color, label)
    % Plot estimation results
    
    k = 1:length(mu_X_est);
    
    % Gaussian process
    subplot(3, 2, 1);
    plot(k, mu_X_est, color, 'DisplayName', label);
    title('Gaussian: Estimated Mean');
    ylabel('\mu_X(k)');
    
    subplot(3, 2, 2);
    plot(k, var_X_est, color, 'DisplayName', label);
    title('Gaussian: Estimated Variance');
    ylabel('\sigma_X^2(k)');
    
    % Uniform process
    subplot(3, 2, 3);
    plot(k, mu_Y_est, color, 'DisplayName', label);
    title('Uniform: Estimated Mean');
    ylabel('\mu_Y(k)');
    
    subplot(3, 2, 4);
    plot(k, var_Y_est, color, 'DisplayName', label);
    title('Uniform: Estimated Variance');
    ylabel('\sigma_Y^2(k)');
    
    % Exponential process
    subplot(3, 2, 5);
    plot(k, mu_Z_est, color, 'DisplayName', label);
    title('Exponential: Estimated Mean');
    xlabel('Sample index k');
    ylabel('\mu_Z(k)');
    
    subplot(3, 2, 6);
    plot(k, var_Z_est, color, 'DisplayName', label);
    title('Exponential: Estimated Variance');
    xlabel('Sample index k');
    ylabel('\sigma_Z^2(k)');
end

function plot_final_results(mu_X_est, var_X_est, mu_Y_est, var_Y_est, mu_Z_est, var_Z_est, ...
                           mu_X_theo, var_X_theo, mu_Y_theo, var_Y_theo, mu_Z_theo, var_Z_theo)
    % Plot final results with theoretical values
    
    figure('Position', [200, 200, 1200, 800]);
    k = 1:length(mu_X_est);
    k_theo = linspace(1, length(mu_X_est), length(mu_X_theo));
    
    % Gaussian process
    subplot(3, 2, 1);
    plot(k, mu_X_est, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimated');
    hold on;
    plot(k_theo, mu_X_theo, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical');
    title('Gaussian Process Mean');
    legend;
    grid on;
    
    subplot(3, 2, 2);
    plot(k, var_X_est, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimated');
    hold on;
    plot(k_theo, var_X_theo, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical');
    title('Gaussian Process Variance');
    legend;
    grid on;
    
    % Uniform process
    subplot(3, 2, 3);
    plot(k, mu_Y_est, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimated');
    hold on;
    plot(k_theo, mu_Y_theo, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical');
    title('Uniform Process Mean');
    legend;
    grid on;
    
    subplot(3, 2, 4);
    plot(k, var_Y_est, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimated');
    hold on;
    plot(k_theo, var_Y_theo, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical');
    title('Uniform Process Variance');
    legend;
    grid on;
    
    % Exponential process
    subplot(3, 2, 5);
    plot(k, mu_Z_est, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimated');
    hold on;
    plot(k_theo, mu_Z_theo, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical');
    title('Exponential Process Mean');
    xlabel('Sample index k');
    legend;
    grid on;
    
    subplot(3, 2, 6);
    plot(k, var_Z_est, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimated');
    hold on;
    plot(k_theo, var_Z_theo, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical');
    title('Exponential Process Variance');
    xlabel('Sample index k');
    legend;
    grid on;
end

function display_results(mu_est, var_est, mu_theo, var_theo, process_name)
    % Display comparison between estimated and theoretical values
    fprintf('\n%s Process:\n', process_name);
    fprintf('Estimated Mean: %.4f ± %.4f (Theoretical: %.4f)\n', ...
            mean(mu_est), std(mu_est), mean(mu_theo));
    fprintf('Estimated Variance: %.4f ± %.4f (Theoretical: %.4f)\n', ...
            mean(var_est), std(var_est), mean(var_theo));
    fprintf('Mean Error: %.4f (%.2f%%)\n', ...
            abs(mean(mu_est) - mean(mu_theo)), ...
            100*abs(mean(mu_est) - mean(mu_theo))/abs(mean(mu_theo)));
    fprintf('Variance Error: %.4f (%.2f%%)\n\n', ...
            abs(mean(var_est) - mean(var_theo)), ...
            100*abs(mean(var_est) - mean(var_theo))/abs(mean(var_theo)));
end

% Run the analysis
random_process_analysis();