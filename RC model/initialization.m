%% Initialization - Supercapacitor model
% The values are taken from Sulabh's PowerPoint presentation.

% I_T DC current
I_T = 10;        % DC current [A]

% RC0 model
R_s = 0.03*5;      % Series resistance [Ohm]
C0 = 88/5;        % Capacitance [F]

% RC branches
R1 = 0.015;       % [Ohm]
C1 = 100;         % [F]

R2 = 0.020;       % [Ohm]
C2 = 100;        % [F]

R3 = 0.030;       % [Ohm]
C3 = 1000;       % [F]

Cvar = 1;        %[omh]
R_leak = 1e-3;      % [ohm]

