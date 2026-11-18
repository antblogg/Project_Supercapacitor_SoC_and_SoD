%% soc_sod_estimation.m
% State of Charge (SoC) and State of Duration (SoD) for the supercapacitor.
%
%   SoC = (V^2 - Vmin^2) / (Vmax^2 - Vmin^2)      (same definition as the lab's "soe" signal)
%   SoD = time [s] until the voltage reaches Vmin at a given discharge power
%
% Compared methods:
%   SoC : ideal capacitor (V0 + Q/C)  vs  physics-based model (C(V) + leakage + ESR)
%   SoD : ideal capacitor formula     vs  physics-based model simulated at constant power
%
% Run compare_models_8files.m first (test mode 'B'): this script then uses the
% physics-model parameters fitted WITHOUT each test file. Otherwise mean values are used.

clear; clc; close all;


%% 1) SETTINGS

dataFolder = '';     % '' = same folder as this script

files = { ...
    'usn_demo_test_processed_short.csv', ...
    'usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_short.csv', ...
    'usn-supercap_green-profile_recharge_2026-09-25_processed_short.csv', ...
    'usn-supercap_red-profile_recharge_2026-09-25_processed_short.csv', ...
    '2026-10-09_fredrik_processed_short.csv', ...
    '2026-10-09_fredrik_2_processed_short.csv', ...
    '2026-10-09_binabdu_v2_processed_short.csv', ...
    '2026-10-09_binabdu_run_1_processed_short.csv'};

names = {'T1 demo','T2 green no-rech','T3 green rech','T4 red rech', ...
         'N1 fredrik','N2 fredrik_2','N3 binabdu_v2','N4 binabdu_run_1'};

Vmax  = 380;     % [V] SoC = 100 %  (lab definition)
Vmin  = 180;     % [V] SoC = 0 %, and SoD cut-off voltage
C_nom = 16;      % [F] ideal capacitor

plotFile = 6;    % file used for the plots


%% 2) LOAD DATA

if isempty(dataFolder)
    dataFolder = fileparts(mfilename('fullpath'));
end

nFiles = numel(files);
data   = struct();

for k = 1:nFiles
    raw = readtable(fullfile(dataFolder, files{k}));
    t   = raw.time - raw.time(1);
    V   = raw.sc_v;
    I   = raw.sc_a .* currentSign(t, V);     % + = charging, - = discharging

    data(k).t = t;
    data(k).V = V;
    data(k).I = I;
    data(k).V_ideal = V(1) + cumtrapz(t, I) / C_nom;
end


%% 3) PHYSICS-MODEL PARAMETERS  [C0, k, Rp, ESR]

params = repmat([12.97, 0.0143, 583, 0.0473], nFiles, 1);   % default: mean values

if isfile(fullfile(dataFolder, 'model_comparison.mat'))
    S = load(fullfile(dataFolder, 'model_comparison.mat'), 'P');
    for k = 1:nFiles
        if any(strcmp(S.P.Properties.RowNames, names{k}))
            row = S.P(names{k}, :);
            params(k,:) = [row.C0_F, row.k_F_per_V, row.Rp_ohm, row.ESR_mOhm/1000];
        end
    end
    disp('Using physics-model parameters from model_comparison.mat (fitted without each test file).');
else
    disp('model_comparison.mat not found: using mean physics-model parameters.');
end

for k = 1:nFiles
    data(k).V_phys = simulateGreyBox(params(k,:), data(k).t, data(k).V, data(k).I);
end


%% 4) STATE OF CHARGE (SoC)

socFun = @(V) (V.^2 - Vmin^2) / (Vmax^2 - Vmin^2) * 100;     % [%]

socRMSE = zeros(nFiles, 2);
for k = 1:nFiles
    SoC_ref   = socFun(data(k).V);          % from measured voltage (reference)
    SoC_ideal = socFun(data(k).V_ideal);    % estimated from current only, ideal capacitor
    SoC_phys  = socFun(data(k).V_phys);     % estimated from current only, physics model

    socRMSE(k,1) = sqrt(mean((SoC_ref - SoC_ideal).^2));
    socRMSE(k,2) = sqrt(mean((SoC_ref - SoC_phys ).^2));

    data(k).SoC_ref = SoC_ref;  data(k).SoC_ideal = SoC_ideal;  data(k).SoC_phys = SoC_phys;
end

socTable = array2table([socRMSE; mean(socRMSE,1)], ...
    'VariableNames', {'Ideal_capacitor','Physics_model'}, 'RowNames', [names, {'MEAN'}]);
disp('===== SoC RMSE [%-points] =====');
disp(socTable)


%% 5) STATE OF DURATION (SoD)

% Find full discharges (from about Vmax down to Vmin) and compare the
% predicted remaining time with the measured remaining time.

sodRows = [];     % [file, event, V, P, measured, ideal, physics]

for k = 1:nFiles
    events = findFullDischarges(data(k).t, data(k).V, data(k).I, Vmin);

    for e = 1:size(events, 1)
        a = events(e,1);  b = events(e,2);
        t = data(k).t;  V = data(k).V;  I = data(k).I;

        P = mean(abs(V(a:b) .* I(a:b)));          % [W] discharge power of this event

        for i = a:10:b                            % every 0.5 s
            measured = t(b) - t(i);
            if measured < 0.5, continue, end

            % ideal capacitor:  SoD = 1/2 * C * (V^2 - Vmin^2) / P
            ideal = 0.5 * C_nom * (V(i)^2 - Vmin^2) / P;

            % physics model: start from internal voltage, discharge at constant P
            Vc0  = V(i) + params(k,4) * abs(I(i));     % remove ESR drop
            phys = sodPhysics(params(k,:), Vc0, P, Vmin);

            sodRows = [sodRows; k, e, V(i), P, measured, ideal, phys]; %#ok<AGROW>
        end
    end
    if ~isempty(events)
        fprintf('%-18s %d full discharges found\n', names{k}, size(events,1));
    end
end

if isempty(sodRows)
    disp('No full discharges found - SoD cannot be validated.');
else
    meas = sodRows(:,5);
    errIdeal = sodRows(:,6) - meas;
    errPhys  = sodRows(:,7) - meas;

    sodTable = table( ...
        [mean(abs(errIdeal)); mean(abs(errPhys))], ...
        [sqrt(mean(errIdeal.^2)); sqrt(mean(errPhys.^2))], ...
        [mean(errIdeal); mean(errPhys)], ...
        [100*mean(abs(errIdeal)./meas); 100*mean(abs(errPhys)./meas)], ...
        'VariableNames', {'MAE_s','RMSE_s','MeanError_s','MAPE_percent'}, ...
        'RowNames', {'Ideal capacitor','Physics model'});
    fprintf('\n===== SoD error (%d points, %.1f kW mean power) =====\n', ...
            numel(meas), mean(sodRows(:,4))/1000);
    disp(sodTable)
end


%% 6) PLOTS

% --- SoC over time ---
k = plotFile;
figure('Name', 'SoC');
plot(data(k).t, data(k).SoC_ref, 'k', 'LineWidth', 1.5); hold on
plot(data(k).t, data(k).SoC_ideal, '--');
plot(data(k).t, data(k).SoC_phys);
grid on
xlabel('Time [s]');  ylabel('SoC [%]');
title(['SoC, test file: ' names{k}], 'Interpreter', 'none');
legend('From measured voltage', 'Ideal capacitor', 'Physics model', 'Location', 'best');

% --- SoD: predicted vs measured ---
if ~isempty(sodRows)
    figure('Name', 'SoD');
    scatter(meas, sodRows(:,6), 8, 'filled'); hold on
    scatter(meas, sodRows(:,7), 8, 'filled');
    plot([0 max(meas)], [0 max(meas)], 'k--');
    grid on
    xlabel('Measured SoD [s]');  ylabel('Predicted SoD [s]');
    title('State of Duration: predicted vs measured');
    legend('Ideal capacitor', 'Physics model', 'Perfect', 'Location', 'best');
end


%% 7) SAVE

save('soc_sod_results.mat', 'socTable', 'sodRows');
writetable(socTable, 'soc_rmse.csv', 'WriteRowNames', true);
if exist('sodTable', 'var')
    writetable(sodTable, 'sod_error.csv', 'WriteRowNames', true);
end


%% ===================== LOCAL FUNCTIONS =====================

function s = currentSign(t, V)
% +1 when the voltage rises (charging), -1 when it falls (discharging).
    Ts = t(2) - t(1);
    dVdt = gradient(movmean(V, round(1/Ts)), t);
    s = ones(size(V));
    current = 1;
    for n = 1:numel(V)
        if abs(dVdt(n)) > 0.3
            current = sign(dVdt(n));
        end
        s(n) = current;
    end
end

function Vt = simulateGreyBox(p, t, V, I)
% dVc/dt = (I - Vc/Rp) / (C0 + k*Vc),   Vt = Vc + ESR*I
    C0 = p(1);  k = p(2);  Rp = p(3);  ESR = p(4);
    Vc = zeros(size(V));
    Vc(1) = V(1) - ESR * I(1);
    for n = 1:numel(t)-1
        dt = t(n+1) - t(n);
        Vc(n+1) = Vc(n) + dt * (I(n) - Vc(n)/Rp) / (C0 + k * max(Vc(n), 0));
    end
    Vt = Vc + ESR * I;
end

function events = findFullDischarges(t, V, I, Vmin)
% Start: voltage >= 370 V and current switches to discharge (< -30 A).
% End  : voltage reaches Vmin + 2 V within 90 s.
    events = [];
    n = numel(V);
    k = 1;
    while k < n
        if V(k) >= 370 && I(k) > -30 && I(k+1) < -30
            j = k + 1;
            while j < n && I(j) < -5 && V(j) > Vmin + 2 && t(j) - t(k) < 90
                j = j + 1;
            end
            if V(j) <= Vmin + 2
                events = [events; k+1, j]; %#ok<AGROW>
            end
            k = j;
        else
            k = k + 1;
        end
    end
end

function sod = sodPhysics(p, Vc, P, Vmin)
% Simulate a constant-power discharge until the terminal voltage reaches Vmin.
    C0 = p(1);  k = p(2);  Rp = p(3);  ESR = p(4);
    dt = 0.05;  sod = 0;
    while sod < 300
        % constant power:  Vt * I = P  with  Vt = Vc - ESR*I
        D = Vc^2 - 4*ESR*P;
        if D < 0, return, end
        I  = (Vc - sqrt(D)) / (2*ESR);     % discharge current [A]
        Vt = Vc - ESR * I;
        if Vt <= Vmin, return, end
        Vc  = Vc + dt * (-I - Vc/Rp) / (C0 + k*Vc);
        sod = sod + dt;
    end
end
