%% compare_models_8files.m
% Compare voltage models for the supercapacitor on 8 lab tests:
%   1. Ideal capacitor        V = V0 + Q/C
%   2. Physics-based model     C(V) = C0 + k*V, leakage resistor Rp, series resistor ESR
%   3. Random Forest (RF)     pure ML
%   4. SVR linear             pure ML
%   5. Physics-based model + RF         RF learns the error of Physics-based model
%   6. Physics-based model+ SVR         SVR learns the error of the Physics-box model
%
% Two ways to test (choose in SETTINGS):
%   'A' : train on the 4 old tests, test on the 4 new tests
%   'B' : leave-one-out: train on 7 files, test on 1, repeat for all 8
%
% Needs: Statistics and Machine Learning Toolbox  without these two  you
% will get error

clear; clc; close all;
rng(42);


%% 1) SETTINGS

% Folder with the 8 csv files. '' = the same folder as this script.
% Example: dataFolder = 'C:\Users\Abdullah\Desktop\FM project\ML model\data';
dataFolder = '';

files = { ...
    'usn_demo_test_processed_short.csv', ...                                   % 1  (= old test 1)
    'usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_short.csv', ... % 2  (= old test 2)
    'usn-supercap_green-profile_recharge_2026-09-25_processed_short.csv', ...   % 3  (= old test 3)
    'usn-supercap_red-profile_recharge_2026-09-25_processed_short.csv', ...     % 4  (= old test 4)
    '2026-10-09_fredrik_processed_short.csv', ...                              % 5  new
    '2026-10-09_fredrik_2_processed_short.csv', ...                            % 6  new
    '2026-10-09_binabdu_v2_processed_short.csv', ...                           % 7  new
    '2026-10-09_binabdu_run_1_processed_short.csv'};                           % 8  new

names = {'T1 demo','T2 green no-rech','T3 green rech','T4 red rech', ...
         'N1 fredrik','N2 fredrik_2','N3 binabdu_v2','N4 binabdu_run_1'};

testMode = 'B';      % 'A' = old 4 -> new 4,   'B' = leave-one-out on all 8
plotFile = 6;        % which test file to plot over time

C_nom   = 16;        % [F] capacitance for the ideal capacitor
nTrees  = 100;       % RF: number of trees
minLeaf = 5;         % RF: minimum samples per leaf
stride  = 10;        % use every 10th training sample (faster)


%% 2) LOAD DATA AND MAKE FEATURES

nFiles = numel(files);
data   = struct();

if isempty(dataFolder)
    dataFolder = fileparts(mfilename('fullpath'));   % folder of this script
end

% --- check that all files are there before starting ---
missing = {};
for k = 1:nFiles
    if ~isfile(fullfile(dataFolder, files{k}))
        missing{end+1} = files{k}; %#ok<SAGROW>
    end
end
if ~isempty(missing)
    fprintf('Looking in folder: %s\n', dataFolder);
    fprintf('Missing file: %s\n', missing{:});
    disp('CSV files found in this folder:');
    found = dir(fullfile(dataFolder, '*.csv'));
    disp({found.name}');
    error('Some data files were not found. Fix dataFolder or the names in SETTINGS.');
end

for k = 1:nFiles

    raw = readtable(fullfile(dataFolder, files{k}));   % comma separated: time, grid_w, sc_v, sc_a, ...
    t   = raw.time - raw.time(1);       % time [s]
    V   = raw.sc_v;                     % stack voltage [V]
    Iabs = raw.sc_a;                    % current MAGNITUDE [A] (always positive in these files)

    % --- give the current a sign: + = charging, - = discharging ---
    % The files only store |I|. The direction is taken from the voltage:
    % voltage going up = charging, going down = discharging.
    I = Iabs .* currentSign(t, V);

    % --- features ---
    Ts = t(2) - t(1);
    data(k).t        = t;
    data(k).V        = V;
    data(k).I        = I;
    data(k).V0       = V(1) * ones(size(V));
    data(k).Ah       = cumtrapz(t, I) / 3600;
    data(k).I_mean10 = movmean(I, [round(10/Ts)-1 0]);
    data(k).I_mean60 = movmean(I, [round(60/Ts)-1 0]);
    data(k).V_ideal  = V(1) + cumtrapz(t, I) / C_nom;

    fprintf('%-18s %6d samples  %5.0f s   V = %5.1f - %5.1f V   I = %6.1f .. %6.1f A\n', ...
            names{k}, numel(V), t(end), min(V), max(V), min(I), max(I));
end


%% 3) WHICH FILES ARE TRAINED / TESTED

if testMode == 'A'
    splits = {struct('train', 1:4, 'test', 5:8)};
else
    splits = cell(1, nFiles);
    for k = 1:nFiles
        splits{k} = struct('train', setdiff(1:nFiles, k), 'test', k);
    end
end

modelNames = {'Ideal cap', 'Physics-based model', 'RF', 'SVR lin', 'Physics-based model+RF', 'Physics-based model+SVR'};
RMSE   = nan(nFiles, numel(modelNames));
params = nan(nFiles, 4);       % 	Physics-based mode parameters used for each test file
saved  = struct();             % predictions for the plot


%% 4) TRAIN AND TEST

for s = 1:numel(splits)
    trainFiles = splits{s}.train;
    testFiles  = splits{s}.test;
    fprintf('\nRun %d of %d: test on %s\n', s, numel(splits), strjoin(names(testFiles), ', '));

    % --- 4a) fit the 	Physics-based model on the training files ---
    p = fitPhysics(data(trainFiles));
    fprintf('   Physics-based model: C0 = %.2f F, k = %.4f F/V, Rp = %.0f ohm, ESR = %.0f mOhm\n', ...
            p(1), p(2), p(3), p(4)*1000);

    for k = 1:nFiles
        data(k).V_Physics = simulatePhysics(p, data(k).t, data(k).V, data(k).I);
    end

    % --- 4b) build training matrices ---
    Xpure = []; XPhysics = []; yV = []; yErr = [];
    for k = trainFiles
        idx   = 1:stride:numel(data(k).V);
        Xpure = [Xpure; data(k).V0(idx), data(k).Ah(idx), data(k).I(idx), data(k).I_mean10(idx), data(k).I_mean60(idx)];
        XPhysics = [XPhysics; data(k).V_Physics(idx), data(k).I(idx), data(k).I_mean10(idx), data(k).I_mean60(idx)];
        yV    = [yV;   data(k).V(idx)];
        yErr  = [yErr; data(k).V(idx) - data(k).V_Physics(idx)];
    end

    % --- 4c) train the ML models ---
    rfPure  = TreeBagger(nTrees, Xpure, yV,   'Method','regression', 'MinLeafSize',minLeaf);
    rfPhysics  = TreeBagger(nTrees, XPhysics, yErr, 'Method','regression', 'MinLeafSize',minLeaf);
    svmPure = fitrsvm(Xpure(1:2:end,:), yV(1:2:end),   'KernelFunction','linear', 'Standardize',true, 'Epsilon',1);
    svmPhysics = fitrsvm(XPhysics(1:2:end,:), yErr(1:2:end), 'KernelFunction','linear', 'Standardize',true, 'Epsilon',1);

    % --- 4d) test ---
    for k = testFiles
        Xp = [data(k).V0, data(k).Ah, data(k).I, data(k).I_mean10, data(k).I_mean60];
        Xg = [data(k).V_Physics, data(k).I, data(k).I_mean10, data(k).I_mean60];

        pred = [data(k).V_ideal, ...
                data(k).V_Physics, ...
                predict(rfPure, Xp), ...
                predict(svmPure, Xp), ...
                data(k).V_Physics + predict(rfPhysics, Xg), ...
                data(k).V_Physics + predict(svmPhysics, Xg)];

        for m = 1:size(pred, 2)
            RMSE(k, m) = sqrt(mean((data(k).V - pred(:, m)).^2));
        end
        params(k, :) = p;

        if k == plotFile
            saved.t = data(k).t;  saved.V = data(k).V;  saved.pred = pred;
        end
    end
end


%% 5) RESULTS TABLE (RMSE in V)

tested   = find(~isnan(RMSE(:,1)));
meanRMSE = mean(RMSE(tested,:), 1);
T = array2table([RMSE(tested,:); meanRMSE], ...
                'VariableNames', matlab.lang.makeValidName(modelNames), ...
                'RowNames', [names(tested), {'MEAN'}]);

fprintf('\n===== RMSE [V], test mode %s =====\n', testMode);
disp(T)

P = array2table([params(tested,1:3), params(tested,4)*1000], ...
    'VariableNames', {'C0_F','k_F_per_V','Rp_ohm','ESR_mOhm'}, 'RowNames', names(tested));
disp('Physics-based model parameters (fitted on the training files):');
disp(P)


%% 6) PLOTS

% --- mean RMSE per model ---
figure('Name', 'Mean RMSE');
bar(meanRMSE);
set(gca, 'XTickLabel', modelNames);
ylabel('Mean RMSE [V]');
title(['Model comparison, test mode ' testMode]);
grid on

% --- voltage over time for one file ---
if isfield(saved, 't')
    figure('Name', 'Voltage prediction');
    plot(saved.t, saved.V, 'k', 'LineWidth', 1.5); hold on
    plot(saved.t, saved.pred(:,1), '--');
    plot(saved.t, saved.pred(:,2));
    plot(saved.t, saved.pred(:,5));
    plot(saved.t, saved.pred(:,4));
    grid on
    xlabel('Time [s]');  ylabel('Voltage [V]');
    title(['Test file: ' names{plotFile}]);
    legend('Measured', 'Ideal cap', 'Physics-based model', 'Physics-based model+RF', 'SVR lin', 'Location', 'best');
end


%% 7) SAVE

save('model_comparison.mat', 'T', 'P', 'modelNames', 'names');
writetable(T, 'model_comparison_rmse.csv', 'WriteRowNames', true);


%% ===================== LOCAL FUNCTIONS =====================

function s = currentSign(t, V)
% +1 when the voltage rises (charging), -1 when it falls (discharging).
% Small slopes keep the previous sign (avoids noise flipping it).
    Ts = t(2) - t(1);
    Vs = movmean(V, round(1/Ts));         % smooth over 1 s
    dVdt = gradient(Vs, t);
    s = ones(size(V));
    current = 1;
    for n = 1:numel(V)
        if abs(dVdt(n)) > 0.3             % [V/s] threshold
            current = sign(dVdt(n));
        end
        s(n) = current;
    end
end

function Vt = simulatePhysics(p, t, V, I)
% Physics-based model capacitor:  dVc/dt = (I - Vc/Rp) / (C0 + k*Vc),  Vt = Vc + ESR*I
    C0 = p(1);  k = p(2);  Rp = p(3);  ESR = p(4);
    Vc = zeros(size(V));
    Vc(1) = V(1) - ESR * I(1);
    for n = 1:numel(t)-1
        dt = t(n+1) - t(n);
        C  = C0 + k * max(Vc(n), 0);
        Vc(n+1) = Vc(n) + dt * (I(n) - Vc(n)/Rp) / C;
    end
    Vt = Vc + ESR * I;
end

function p = fitPhysics(trainData)
% Find C0, k, Rp, ESR that give the lowest voltage error on the training files.
% Parameters are scaled so fminsearch treats them equally.
    scale = [1, 0.01, 100, 0.01];               % C0 [F], k [F/V], Rp [ohm], ESR [ohm]
    p0    = [13.6, 0.014, 640, 0.045] ./ scale; % start guess
    cost  = @(q) PhysicsCost(q .* scale, trainData);
    opts  = optimset('MaxIter', 300, 'Display', 'off');
    p     = fminsearch(cost, p0, opts) .* scale;
end

function J = PhysicsCost(p, trainData)
    if p(1) < 1 || p(3) < 10 || p(4) < 0
        J = 1e9;  return
    end
    J = 0;
    for k = 1:numel(trainData)
        idx = 1:10:numel(trainData(k).V);        % every 10th sample (0.5 s) for speed
        Vt  = simulatePhysics(p, trainData(k).t(idx), trainData(k).V(idx), trainData(k).I(idx));
        J   = J + sqrt(mean((trainData(k).V(idx) - Vt).^2));
    end
    J = J / numel(trainData);
end
