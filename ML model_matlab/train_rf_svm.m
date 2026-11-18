%% train_rf_svm.m
% Predict supercapacitor terminal voltage V with Random Forest (RF) and
% Support Vector Regression (SVR), and compare with an ideal capacitor model.
%
% Data : 4 test files (test 1-4.csv), 3 used for training, 1 for testing.
% Needs: Statistics and Machine Learning Toolbox (TreeBagger, fitrsvm).

clear; clc; close all;
rng(42);   % same random numbers every run -> same result every run


%% 1) SETTINGS  (change things here)

files    = {'test 1.csv', 'test 2.csv', 'test 3.csv', 'test 4.csv'};
testFile = 1;        % which file is the test set (try 1, 2, 3 and 4)

C_nom    = 16;       % [F]  stack capacitance (fitted from the lab data)
hybrid   = false;     % true : ML learns only the ERROR of the ideal capacitor
                     % false: ML predicts the full voltage on its own

nTrees   = 100;      % RF: number of trees
minLeaf  = 5;        % RF: minimum samples per leaf
stride   = 10;       % use every 10th training sample (faster, less duplicates)


%% 2) LOAD DATA AND MAKE FEATURES

allData = table();

for k = 1:numel(files)

    % --- read file (semicolon separated: ts ; V_sc_V ; I_sc_A) ---
    raw = readtable(files{k}, 'Delimiter', ';', 'VariableNamingRule', 'preserve');

    t = raw.ts - raw.ts(1);   % time [s], starting at 0
    V = raw.V_sc_V;           % stack voltage [V]
    I = -raw.I_sc_A;          % current [A]. Sign flipped so that charging = positive
                              % (in the files the voltage rises when I is negative)

    Ts = t(2) - t(1);         % sample time [s] (0.05 s)

    % --- features (inputs to the ML models) ---
    V0       = V(1) * ones(size(V));                 % start voltage of the test
    Ah       = cumtrapz(t, I) / 3600;                % charge moved in/out [Ah]
    I_mean10 = movmean(I, [round(10/Ts)-1 0]);       % mean current, last 10 s
    I_mean60 = movmean(I, [round(60/Ts)-1 0]);       % mean current, last 60 s

    % --- ideal capacitor model:  V = V0 + Q / C ---
    V_phys = V(1) + cumtrapz(t, I) / C_nom;

    % --- store this file in one big table ---
    fileID = k * ones(size(V));
    T = table(fileID, t, V, I, V0, Ah, I_mean10, I_mean60, V_phys);
    allData = [allData; T];

    fprintf('%s: %d samples, %.0f s, V = %.0f-%.0f V\n', ...
            files{k}, numel(V), t(end), min(V), max(V));
end


%% 3) CHOOSE INPUTS AND SPLIT INTO TRAIN / TEST

if hybrid
    inputs = {'V_phys', 'I', 'I_mean10', 'I_mean60'};
else
    inputs = {'V0', 'Ah', 'I', 'I_mean10', 'I_mean60'};
end

trainData = allData(allData.fileID ~= testFile, :);
testData  = allData(allData.fileID == testFile, :);

trainData = trainData(1:stride:end, :);    % thin out the training data

Xtrain = trainData{:, inputs};
Xtest  = testData{:, inputs};

% What the ML models learn (the "target"):
if hybrid
    yTrain = trainData.V - trainData.V_phys;   % error of the ideal capacitor
else
    yTrain = trainData.V;                      % the full voltage
end

fprintf('\nTest file: %s | training samples: %d | test samples: %d\n\n', ...
        files{testFile}, height(trainData), height(testData));


%% 4) TRAIN RANDOM FOREST

rfModel = TreeBagger(nTrees, Xtrain, yTrain, ...
                     'Method', 'regression', ...
                     'MinLeafSize', minLeaf, ...
                     'OOBPrediction', 'on', ...
                     'OOBPredictorImportance', 'on', ...
                     'PredictorNames', inputs);


%% 5) TRAIN SUPPORT VECTOR REGRESSION (SVR)

% Linear kernel: simple straight-line model, can extrapolate
svmLinear = fitrsvm(Xtrain, yTrain, ...
                    'KernelFunction', 'linear', ...
                    'Standardize', true, ...
                    'Epsilon', 1);

% Gaussian (RBF) kernel: curved model, cannot extrapolate well
svmRBF = fitrsvm(Xtrain, yTrain, ...
                 'KernelFunction', 'gaussian', ...
                 'KernelScale', 'auto', ...
                 'BoxConstraint', 100, ...
                 'Standardize', true, ...
                 'Epsilon', 1);


%% 6) PREDICT ON THE TEST FILE

V_true  = testData.V;
V_ideal = testData.V_phys;        % ideal capacitor alone

if hybrid
    V_RF     = V_ideal + predict(rfModel,   Xtest);
    V_SVMlin = V_ideal + predict(svmLinear, Xtest);
    V_SVMrbf = V_ideal + predict(svmRBF,    Xtest);
else
    V_RF     = predict(rfModel,   Xtest);
    V_SVMlin = predict(svmLinear, Xtest);
    V_SVMrbf = predict(svmRBF,    Xtest);
end


%% 7) RESULTS TABLE  (RMSE, MAE, R2)

models    = {'Ideal capacitor', 'Random Forest', 'SVR linear', 'SVR RBF'};
predicted = [V_ideal, V_RF, V_SVMlin, V_SVMrbf];

RMSE = zeros(4,1);  MAE = zeros(4,1);  R2 = zeros(4,1);
for m = 1:4
    err     = V_true - predicted(:, m);
    RMSE(m) = sqrt(mean(err.^2));
    MAE(m)  = mean(abs(err));
    R2(m)   = 1 - sum(err.^2) / sum((V_true - mean(V_true)).^2);
end

results = table(RMSE, MAE, R2, 'RowNames', models);
disp(results)


%% 8) PLOTS

t_test = testData.t;

% --- Figure 1: voltage over time ---
figure('Name', 'Voltage prediction');
plot(t_test, V_true,   'k',  'LineWidth', 1.5); hold on
plot(t_test, V_ideal,  '--', 'LineWidth', 1);
plot(t_test, V_RF,     'LineWidth', 1);
plot(t_test, V_SVMlin, 'LineWidth', 1);
plot(t_test, V_SVMrbf, 'LineWidth', 1);
grid on
xlabel('Time [s]');  ylabel('Voltage [V]');
title(['Test file: ' files{testFile}]);
legend('Measured', 'Ideal capacitor', 'Random Forest', 'SVR linear', 'SVR RBF', ...
       'Location', 'best');

% --- Figure 2: predicted vs measured ---
figure('Name', 'Predicted vs measured');
scatter(V_true, V_RF,     4, 'filled'); hold on
scatter(V_true, V_SVMlin, 4, 'filled');
plot([min(V_true) max(V_true)], [min(V_true) max(V_true)], 'k--');
grid on
xlabel('Measured V [V]');  ylabel('Predicted V [V]');
legend('Random Forest', 'SVR linear', 'Perfect fit', 'Location', 'best');

% --- Figure 3: RF details ---
figure('Name', 'Random Forest details');
subplot(1,2,1)
plot(oobError(rfModel)); grid on
xlabel('Number of trees');  ylabel('OOB error (MSE)');
title('OOB error');

subplot(1,2,2)
bar(rfModel.OOBPermutedPredictorDeltaError); grid on
set(gca, 'XTickLabel', inputs, 'TickLabelInterpreter', 'none');
ylabel('Importance');
title('Input importance');


%% 9) SAVE

save('rf_svm_models.mat', 'rfModel', 'svmLinear', 'svmRBF', 'inputs', 'results');
