%% Parameter fitting - Supercapacitor RC model
% Nonlinear least-squares parameter estimation.
% This code was written with the help of AI (ChatGPT).

clear
clc
close all

%% Initial parameter values
% Model name
Model_name = 'RC_model_elc_lib';

% Datasheet parameters
R_s = 7.6e-3*5;      % Series resistance
C0  = 88/5;        % Main capacitance 


% % Parameters that can be optimized
R1 = 100;        % [Ohm]
C1 = 100;       % [F]

R2 = 100;        % [Ohm]
C2 = 1000;      % [F]

R3 = 100;        % [Ohm]
C3 = 1000;      % [F]
R_leak = 800;   % [Ohm]
Cvar   = 3;     % [F]


%% Select parameters to optimize

%active_parameters = { 'R_leak','R1','C1','R2','C2','Cvar'}; % Literature

%active_parameters = { 'R1','C1','R2','C2','R3','C3','R_leak'}; % RC3
active_parameters = { R1','C1','R2','C2','R_leak','C0', 'R_s'}; % RC2  
%active_parameters = { 'R1','C1','R_leak'}; % RC1

%active_parameters = { 'R1','C1','R2','C2','R3','C3'}; % RC3
%active_parameters = { 'R1','C1','R2','C2'}; % RC2  
%active_parameters = { 'R1','C1'}; % RC1


%% Load measurement data from CSV

filename = "C:\Users\...\2026-10-09_fredrik_2_processed_full.csv";

% Read CSV file
data = readtable(filename, ...
    'VariableNamingRule', 'preserve');


%% Select measurement columns

time    = data.time;          % Time [s]
current = data.sc_a_signed;      % Supercapacitor current [A]
voltage = data.sc_v;      % Supercapacitor voltage [V]

% Convert to column vectors
time    = double(time(:));
current = double(current(:));
voltage = double(voltage(:));


%% Remove invalid measurement values

valid = ...
    isfinite(time) & ...
    isfinite(current) & ...
    isfinite(voltage);

time    =  time(valid);
current = -current(valid);
voltage =  voltage(valid);


%% Display data information

fprintf('\nMeasurement data loaded:\n')
fprintf('Number of samples = %d\n', length(time))
fprintf('Start time        = %.3f s\n', time(1))
fprintf('End time          = %.3f s\n', time(end))

fprintf('\nCurrent:\n')
fprintf('Start current     = %.3f A\n', current(1))
fprintf('Mean current      = %.3f A\n', mean(current))
fprintf('Min current       = %.3f A\n', min(current))
fprintf('Max current       = %.3f A\n', max(current))

fprintf('\nVoltage:\n')
fprintf('Start voltage     = %.3f V\n', voltage(1))
fprintf('End voltage       = %.3f V\n', voltage(end))
fprintf('Min voltage       = %.3f V\n', min(voltage))
fprintf('Max voltage       = %.3f V\n', max(voltage))


%% Initial supercapacitor voltage

% The model should start at the same voltage as the measurement.

V_initial = voltage(1);

%% Create current input for Simulink

I_T = [time current];

%% All available parameters

parameter_names = {'R1', 'C1', 'R2', 'C2', 'R3', 'C3','R_leak', 'Cvar','C0','Ini_fac1', 'Ini_fac2', 'Ini_fac3','Ini_fac4','k','R_leak_var', 'R_s', 'Hvar'}; 

parameter_values = [ R1, C1, R2, C2, R3, C3, R_leak, Cvar, C0, Ini_fac1, Ini_fac2, Ini_fac3, Ini_fac4 ,k ,R_leak_var, R_s, Hvar];


%% Parameter limits

lower_bounds = [ ...

    1e-3 , 1e-6...         % R1, C1
    1e-3 , 1e-6 ...        % R2, C2
    1e-3 , 1e-6 ...        % R3, C3
    1, 1e-6 ...            % R_leak, Cvar
    1e-3 ...               % C0
    0, 0, 0,0 ...          % Ini_fac1, Ini_fac2, Ini_fac3
    0, 0.001, 1e-6, 1e-6]; %k, R_leak_var, R_s, Hvar

upper_bounds = [ ...

    1e3   , 1e6 ...        % R1, C1
    1e3   , 1e6 ...        % R2, C2
    1e3  , 1e6 ...         % R3, C3
    1e6   , 100 ...        % R_leak, Cvar
    100 ...                % C0
    3, 3, 3,3 ...          % Ini_fac1, Ini_fac2, Ini_fac3
    10, 2000, 1e6, 1e6];   %k, R_leak_var, R_s, Hvar


%% Select active parameters

use_parameter = ismember( parameter_names, active_parameters); % Checking which parameters are selected for optimization.

active_names = ...
    parameter_names(use_parameter);

p0 = parameter_values(use_parameter);

lb = lower_bounds(use_parameter);

ub = upper_bounds(use_parameter);


%% Send all parameter values to Simulink

for i = 1:length(parameter_names)

    assignin( ...
        'base', ...
        parameter_names{i}, ...
        parameter_values(i));

end

%% Load Simulink model

% Only ONE model/subsystem can be active.
load_system(Model_name)


%% Set simulation stop time

% Simulation ends at the same time as the measurement data.

set_param(  Model_name,"StopTime",num2str(time(end)));


%% Display starting values

fprintf('\nStarting parameters:\n\n')

for i = 1:length(p0)

    fprintf( ...
        '%s = %.6g\n', ...
        active_names{i}, ...
        p0(i));

end

%% File for optimization history

history_file = 'optimization_history.csv';

% Delete old file
if isfile(history_file)
    delete(history_file)
end
%% Optimization settings

options = optimoptions( ...
    'lsqnonlin', ...
    'Display', 'iter', ...
    'TypicalX', p0, ...
    'FiniteDifferenceType', 'central', ...
    'FiniteDifferenceStepSize', 0.01, ...
    'MaxIterations', 20, ...
    'MaxFunctionEvaluations', 500, ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-10, ...
    'OutputFcn', @(p,optimValues,state) ...
        save_parameter_history(p, optimValues, state, ...
        active_names, history_file));

%% Run parameter optimization Alt1

p_opt = lsqnonlin( ...
    @(p) objective_function( p, active_names, time,  voltage, Model_name), ...
    p0, ...
    lb, ...
    ub, ...
    options);

%% Display optimized parameters

fprintf('\nFixed parameters:\n\n')

fprintf('R_s = %.6f Ohm\n', R_s)
fprintf('C0  = %.2f F\n', C0)


fprintf('\nOptimized parameters:\n\n')

for i = 1:length(p_opt)

    fprintf( ...
        '%s = %.6g\n', ...
        active_names{i}, ...
        p_opt(i));

end

%% Send optimized parameters to Simulink

for i = 1:length(p_opt)

    assignin( ...
        'base', ...
        active_names{i}, ...
        p_opt(i));

end

%% Run final simulation

simOut = sim(Model_name);


%% Get simulated terminal voltage

V_sim = simOut.V_T; %V_T = terminal voltage


%% Interpolate simulated voltage

% Calculate simulated voltage at the same time points
% as the measured voltage.

V_estimated = interp1(V_sim.Time, V_sim.Data, time);

%% Calculate voltage error

voltage_error = V_estimated - voltage;

%% Calculate RMSE

RMSE = sqrt(mean(voltage_error.^2));

fprintf( ...
    '\nVoltage RMSE = %.6f V\n', ...
    RMSE);

%% Plot measured and estimated voltage

figure()

plot( ...
    time, ...
    voltage, ...
    'DisplayName', ...
    'Measured voltage')

hold on

plot( ...
    time, ...
    V_estimated, ...
    'DisplayName', ...
    'Estimated voltage')

xlabel('Time [s]')
ylabel('Voltage [V]')

title('Measured and estimated voltage')

legend
grid on

%% Save results to CSV

% Folder for saving results
results_folder = ...
    'C:\...\Project\Results';

% Create folder if needed
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

% Results at measurement time points
results = table( ...
    time, ...
    voltage, ...
    V_estimated, ...
    'VariableNames', {'Time', 'V_measured', 'V_T'});

% Add total RMSE
results.RMSE = repmat(RMSE, height(results), 1);

% Add fixed parameters
results.R_s = repmat(R_s, height(results), 1);
results.C0  = repmat(C0,  height(results), 1);

% Add optimized parameters
for i = 1:length(p_opt)

    results.(active_names{i}) = ...
        repmat(p_opt(i), height(results), 1);

end

% Save CSV file
csv_file = fullfile( ...
    results_folder, ...
    'optimized_results.csv');

writetable(results, csv_file);

% Save voltage plot as MATLAB figure
plot_file = fullfile( ...
    results_folder, ...
    'voltage_comparison.fig');

savefig(figure(1), plot_file);
%% Objective function 

function error = objective_function( p, active_names, time, voltage, Model_name)

    %% Send active parameters to Simulink

    for i = 1:length(p)

        assignin( 'base', active_names{i}, p(i));
    end


    %% Run Simulink model

    simOut = sim(Model_name);


    %% Get simulated terminal voltage

    V_sim = simOut.V_T;


    %% Match simulation and measurement time

    V_model = interp1( ...
        V_sim.Time, ...
        V_sim.Data, ...
        time);


    %% Check simulated values

    if any(~isfinite(V_model))

        error('Simulated voltage contains NaN or Inf values.')

    end


    %% Calculate voltage error

    error = V_model - voltage;


end

function stop = save_parameter_history( ...
    p, optimValues, state, active_names, history_file)

    stop = false;

    % Save starting values
    if strcmp(state, 'init')

        values = [optimValues.iteration, p(:)'];

        names = [{'Iteration'}, active_names];

        T = array2table(values, ...
            'VariableNames', names);

        writetable(T, history_file);

    end

    % Save each optimization iteration
    if strcmp(state, 'iter')

        values = [optimValues.iteration, p(:)'];

        names = [{'Iteration'}, active_names];

        T = array2table(values, ...
            'VariableNames', names);

        writetable(T, history_file, ...
            'WriteMode', 'append', ...
            'WriteVariableNames', false);

    end
end