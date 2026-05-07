%% Hydraulic Transient Dynamics - Transmission Line Model
%
%  Given Parameters
%  -----------------------------------------------------------------------
%  Subsystem          | Quantity                    | Symbol | Value
%  -----------------------------------------------------------------------
%  Hydraulic supply   | Supply pressure (gauge)     | P_s    | 13.8e6 Pa
%  Hydraulic supply   | Tank pressure (gauge)       | P_T    | 0 Pa
%  Fluid              | Density                     | rho    | 850 kg/m^3
%  Fluid              | Dynamic viscosity           | mu     | 0.040 Pa*s
%  Fluid              | Effective bulk modulus       | beta   | 1.5e9 Pa
%  Valve              | Spool time constant         | tau_v  | 0.03 s
%  Valve              | Maximum flow rate           | Q_max  | 2.5e-4 m^3/s
%  Hose               | Length                      | L_h    | 4.0 m
%  Hose               | Inner diameter              | D_h    | 6.0e-3 m
%  Cylinder            | Bore diameter               | D_c    | 50e-3 m
%  Load               | Equivalent moving mass      | m      | 1200 kg
%  Load               | Viscous damping             | b      | 1800 N*s/m
%  Load               | Spring stiffness            | k_cyl  | 9.0e4 N/m
%  Design spec        | Steady-state stroke at P_s  | x_max  | 0.3048 m
%  Orientation        | Horizontal (no gravity)     | --     | --
%  -----------------------------------------------------------------------
%
%  Equations
%  -----------------------------------------------------------------------
%  Eq. 1  | Hose cross-section area   | A_hose  = pi * D_h^2 / 4
%  Eq. 2  | Hose internal volume      | V_hose  = A_hose * L_h
%  Eq. 3  | Cylinder piston area      | A_c     = pi * D_c^2 / 4
%  Eq. 4  | Hose resistance (laminar) | R       = 128 * mu * L_h / (pi * D_h^4)
%  Eq. 5  | Hose capacitance          | C       = V_hose / beta
%  Eq. 6  | Hose inertance            | I       = rho * L_h / A_hose
%  Eq. 7  | Line natural frequency    | omega_n = 1 / sqrt(C * I)
%  Eq. 8  | Line damping ratio        | zeta    = (R / 2) * sqrt(C / I)
%  Eq. 9  | Line 1% settling time     | t_settle= 5 / (zeta * omega_n)
%  Eq. 10 | Step input                | U(t)    = 1 for t >= 0
%  -----------------------------------------------------------------------
%% Lumped Parameters
%  -----------------------------------------------------------------------
%  Lumped Parameter         | Equation
%  -----------------------------------------------------------------------
%  R (Laminar Assumption)   | R = 128 * mu * L / (pi * D_hose^4)
%  C                        | C = V_hose / beta
%  I                        | I = rho * L / A_hose
%  -----------------------------------------------------------------------
%% Transfer Functions
%  -----------------------------------------------------------------------
%  G1(s) = (m*s^2 + b*s + k_cyl) / [A_c^2 * s * (C*I*s^2 + R*C*s + 1)]
%  G2(s) = C*(m*s^2 + b*s + k_cyl) / [A_c^2 * (C*I*s^2 + R*C*s + 1)]
%  -----------------------------------------------------------------------

%% Given Parameters
P_s    = 13.8e6;        % Supply pressure (gauge)            [Pa]
P_T    = 0;             % Tank pressure (gauge)              [Pa]
rho    = 850;           % Fluid density                      [kg/m^3]
mu     = 0.040;         % Dynamic viscosity                  [Pa*s]
beta   = 1.5e9;         % Effective bulk modulus              [Pa]
tau_v  = 0.03;          % Valve spool time constant           [s]
Q_max  = 2.5e-4;        % Valve maximum flow rate             [m^3/s]
L_h    = 4.0;           % Hose length                        [m]
D_h    = 6.0e-3;        % Hose inner diameter                [m]
D_c    = 50e-3;         % Cylinder bore diameter             [m]
m      = 1200;          % Equivalent moving mass             [kg]
b      = 1800;          % Viscous damping                    [N*s/m]
k_cyl  = 9.0e4;         % Spring stiffness                   [N/m]
x_max  = 0.3048;        % Steady-state stroke at P_s         [m]

%% Equations

% Eq. 1 - Hose cross-section area [m^2]
A_hose = pi * D_h^2 / 4;

% Eq. 2 - Hose internal volume [m^3]
V_hose = A_hose * L_h;

% Eq. 3 - Cylinder piston area [m^2]
A_c = pi * D_c^2 / 4;

% Eq. 4 - Hose resistance (laminar) [Pa*s/m^3]
R = 128 * mu * L_h / (pi * D_h^4);

% Eq. 5 - Hose capacitance [m^3/Pa]
C = V_hose / beta;

% Eq. 6 - Hose inertance [kg/m^4]
I = rho * L_h / A_hose;

% Eq. 7 - Line natural frequency [rad/s]
omega_n = 1 / sqrt(C * I);

% Eq. 8 - Line damping ratio [dimensionless]
zeta = (R / 2) * sqrt(C / I);

% Eq. 9 - Line 1% settling time [s]
t_settle = 5 / (zeta * omega_n);


%% Display computed values
fprintf('--- Computed Line Parameters ---\n');
fprintf('A_hose   = %.6e m^2\n',   A_hose);
fprintf('V_hose   = %.6e m^3\n',   V_hose);
fprintf('A_c      = %.6e m^2\n',   A_c);
fprintf('R        = %.6e Pa*s/m^3\n', R);
fprintf('C        = %.6e m^3/Pa\n', C);
fprintf('I      = %.6e kg/m^4\n', I);
fprintf('omega_n  = %.4f rad/s\n',  omega_n);
fprintf('zeta     = %.6f\n',        zeta);
fprintf('t_settle = %.4f s\n',      t_settle);


%% Transfer Functions
% Common numerator polynomial: m*s^2 + b*s + k_cyl
num_common = [m  b  k_cyl];

% G1(s) - Forward transfer function
%   Numerator:   m*s^2 + b*s + k_cyl
%   Denominator: A_c^2 * s * (C*I*s^2 + R*C*s + 1)
%                = A_c^2 * [C*I*s^3 + R*C*s^2 + s]
G1_num = num_common;
G1_den = A_c^2 * [C*I, R*C, 1, 0];
G1 = tf(G1_num, G1_den);

% G2(s) - Feedback transfer function
%   Numerator:   C * (m*s^2 + b*s + k_cyl)
%   Denominator: A_c^2 * (C*I*s^2 + R*C*s + 1)
G2_num = C * num_common;
G2_den = A_c^2 * [C*I, R*C, 1];

G2 = tf(G2_num, G2_den);
fprintf('\n--- Transfer Functions ---\n');
fprintf('G1(s) = Forward TF:\n');
G1
fprintf('G2(s) = Feedback TF:\n');
G2

%% Pole-Zero Analysis
fprintf('\n===================================================================\n');
fprintf('                     POLE-ZERO ANALYSIS                          \n');
fprintf('===================================================================\n');
 
% --- G1(s) ---
p1 = pole(G1);
z1 = zero(G1);
 
fprintf('\n--- G1(s) Forward TF ---\n');
fprintf('%-6s  %-30s  %-10s  %s\n', 'Index', 'Pole', 'Re(pole)', 'Stability');
fprintf('------  ------------------------------  ----------  ----------\n');
for k = 1:length(p1)
    if imag(p1(k)) == 0
        pole_str = sprintf('%.6e', p1(k));
    else
        pole_str = sprintf('%.6e %+.6ei', real(p1(k)), imag(p1(k)));
    end
    if real(p1(k)) < 0
        stab = 'STABLE';
    elseif real(p1(k)) == 0
        stab = 'MARGINAL';
    else
        stab = 'UNSTABLE';
    end
    fprintf('%-6d  %-30s  %-10.4e  %s\n', k, pole_str, real(p1(k)), stab);
end
 
fprintf('\n%-6s  %-30s\n', 'Index', 'Zero');
fprintf('------  ------------------------------\n');
if isempty(z1)
    fprintf('        (none)\n');
else
    for k = 1:length(z1)
        if imag(z1(k)) == 0
            zero_str = sprintf('%.6e', z1(k));
        else
            zero_str = sprintf('%.6e %+.6ei', real(z1(k)), imag(z1(k)));
        end
        fprintf('%-6d  %-30s\n', k, zero_str);
    end
end
 
% --- G2(s) ---
p2 = pole(G2);
z2 = zero(G2);
 
fprintf('\n--- G2(s) Feedback TF ---\n');
fprintf('%-6s  %-30s  %-10s  %s\n', 'Index', 'Pole', 'Re(pole)', 'Stability');
fprintf('------  ------------------------------  ----------  ----------\n');
for k = 1:length(p2)
    if imag(p2(k)) == 0
        pole_str = sprintf('%.6e', p2(k));
    else
        pole_str = sprintf('%.6e %+.6ei', real(p2(k)), imag(p2(k)));
    end
    if real(p2(k)) < 0
        stab = 'STABLE';
    elseif real(p2(k)) == 0
        stab = 'MARGINAL';
    else
        stab = 'UNSTABLE';
    end
    fprintf('%-6d  %-30s  %-10.4e  %s\n', k, pole_str, real(p2(k)), stab);
end
 
fprintf('\n%-6s  %-30s\n', 'Index', 'Zero');
fprintf('------  ------------------------------\n');
if isempty(z2)
    fprintf('        (none)\n');
else
    for k = 1:length(z2)
        if imag(z2(k)) == 0
            zero_str = sprintf('%.6e', z2(k));
        else
            zero_str = sprintf('%.6e %+.6ei', real(z2(k)), imag(z2(k)));
        end
        fprintf('%-6d  %-30s\n', k, zero_str);
    end
end

%% Simulation work

% Closed loop TF
G_cl = G1 / (1 + G2);
[num_cl, den_cl] = tfdata(G_cl, 'v');
% Simulation Parameters
N_points = 20;
dt = 2*pi / (10 * N_points * omega_n);
t_final = 10;

% Configure and Run Simulink Model
mdl = 'problem1_block_cl';  % your .slx filename without extension
open_system(mdl);

set_param(mdl, 'Solver', 'ode4');
set_param(mdl, 'FixedStep', num2str(dt));
set_param(mdl, 'StopTime', num2str(t_final));

simOut = sim(mdl);

% Extract Logged Signals
t     = simOut.tout;
P_out = simOut.P_out.Data;
Q_in  = simOut.Q_in.Data;
x_cyl = simOut.x_cyl.Data;

% Plot Results
figure;

subplot(3,1,1);
plot(t, P_out / 1e6,'LineWidth',3);
xlabel('Time [s]'); ylabel('P_{out} [MPa]');
title('Cylinder Port Pressure');
grid on;

subplot(3,1,2);
plot(t, Q_in * 1e3 * 60,'LineWidth',3);  % convert m^3/s to L/min
xlabel('Time [s]'); ylabel('Q_{in} [L/min]');
title('Valve Flow into Transmission Line');
grid on;

subplot(3,1,3);
plot(t, x_cyl * 1e3,'LineWidth',3);  % convert m to mm
xlabel('Time [s]'); ylabel('x [mm]');
title('Cylinder Displacement');
grid on;

%% zoom in on transients
figure;
temp = ceil(5*t_settle/dt);
subplot(3,1,1);
plot(t(1:temp), P_out(1:temp) / 1e6,'LineWidth',3);
xlabel('Time [s]'); ylabel('P_{out} [MPa]');
title('Cylinder Port Pressure');
grid on;

subplot(3,1,2);
plot(t(1:temp), Q_in(1:temp) * 1e3 * 60,'LineWidth',3);  % convert m^3/s to L/min
xlabel('Time [s]'); ylabel('Q_{in} [L/min]');
title('Valve Flow into Transmission Line');
grid on;

subplot(3,1,3);
plot(t(1:temp), x_cyl(1:temp) * 1e3,'LineWidth',3);  % convert m to mm
xlabel('Time [s]'); ylabel('x [mm]');
title('Cylinder Displacement');
grid on;


