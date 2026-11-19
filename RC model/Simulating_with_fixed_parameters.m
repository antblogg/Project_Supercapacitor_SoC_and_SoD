%% Simulation with fixed parameters - Supercapacitor RC model

% Simulates the RC model using fixed parameter values.
% This code was written with the help of AI (ChatGPT).

clear
clc
close all

%% Fixed parameter values

Model_name = 'RC_model_elc_lib';

% Datasheet parameters
R_s = 7.6e-3*5;      % Series resistance 
C0 = 88/5;           % Main capacitance 

% RC model parameters
R1 = 889.625;
C1 = 7.02122;
R_leak = 1000.16;



% Store names of defined parameters
parameter_names = who;

%% Load measurement data from CSV

filename = "C:\Users\...\usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_full.csv"; 
%filename = "C:\Users\...\usn-supercap_green-profile_recharge_2026-09-25_processed_full.csv";" 
%filename = "C:\Users\...\usn-supercap_red-profile_recharge_2026-09-25_processed_full.csv";
%filename = "C:\Users\...\usn_demo_test_processed_full.csv";

data = readtable(filename, 'VariableNamingRule', 'preserve');

% Measurement columns
time = double(data.time(:));
current = double(data.sc_a_signed(:));
voltage = double(data.sc_v(:));

% Remove invalid values
valid = isfinite(time) & isfinite(current) & isfinite(voltage);

time = time(valid);
current = -current(valid);
voltage = voltage(valid);

% Sort by time
[time, idx] = sort(time);
current = current(idx);
voltage = voltage(idx);

%% Simulate RC model

% Initial voltage
V_initial = voltage(1);

% Current input
I_T = [time current];

% Load model and set simulation time
load_system(Model_name)
set_param(Model_name, 'StopTime', num2str(time(end)));

% Run simulation
simOut = sim(Model_name);

% Simulated voltage
V_sim = simOut.V_T;

% Interpolate to measurement time
V_estimated = interp1(V_sim.Time, V_sim.Data, time, 'linear');

%% Calculate RMSE

valid_rmse = isfinite(V_estimated) & isfinite(voltage);

RMSE = sqrt(mean((V_estimated(valid_rmse) - voltage(valid_rmse)).^2));

fprintf('\nVoltage RMSE = %.6f V\n', RMSE);

%% Plot measured and simulated voltage

fig_voltage = figure;

plot(time, voltage, 'DisplayName', 'Measured voltage')
hold on
plot(time, V_estimated, 'DisplayName', 'Simulated voltage')

xlabel('Time [s]')
ylabel('Voltage [V]')
title('Measured and simulated voltage')
legend
grid on


%% Save results

% Results folder
folder = "C:\...\Project\Results";

if ~isfolder(folder)
    mkdir(folder);
end

% Create results table
results = table(time, current, voltage, V_estimated, ...
    'VariableNames', {'Time_s', 'Current_A', 'V_measured_V', 'V_estimated_V'});

% Add RMSE
results.RMSE = repmat(RMSE, height(results), 1);

% Add all defined model parameters
for k = 1:length(parameter_names)

    name = parameter_names{k};
    value = eval(name);

    if isnumeric(value) && isscalar(value)
        results.(name) = repmat(value, height(results), 1);
    end

end

% Save results CSV
writetable(results, fullfile(folder, "simulation_results.csv"));

% Save original Simulink voltage
V_T = table(V_sim.Time(:), V_sim.Data(:), ...
    'VariableNames', {'Time_s', 'V_T_V'});

writetable(V_T, fullfile(folder, "V_T.csv"));

% Save MATLAB figures
savefig(fig_voltage, fullfile(folder, "voltage.fig"));

fprintf('Results saved to: %s\n', folder);