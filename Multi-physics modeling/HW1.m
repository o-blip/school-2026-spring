%% Piero Risi Mortola
% Assignment 1 - MECE 6397
%
% Axial SDOF beam IC response (lumped model) + comparison plot
% Model: xddot + (B/M) xdot + (K/M) x = 0
% ICs: x(0)=0.2 m, xdot(0)=0 m/s

clear; clc; close all;

%% Given beam properties
E   = 200e9;        % Pa (N/m^2)
A   = 2.5e-3;       % m^2
L   = 1.0;          % m
rho = 7900;         % kg/m^3

%% Lumped parameters for axial vibration (uniform bar, fixed-free)
K = E*A/L;                  % N/m
M = (1/3)*rho*A*L;          % kg  (effective axial modal mass)

%% Identified damping from data
zeta = 0.046;               % (-) estimated damping ratio
wn   = sqrt(K/M);           % rad/s (undamped natural frequency)
B    = 2*zeta*wn*M;         % N*s/m (viscous damping coefficient)

%% Initial conditions
x0  = 0.2;                  % m
v0  = 0.0;                  % m/s
xIC = [x0; v0];

%% Simulation time
tEnd = 0.01;                % s
tspan = [0 tEnd];

%% ODE definition (free response: F(t)=0)
ode = @(t, x) [ ...
    x(2); ...
    -(B/M)*x(2) - (K/M)*x(1) ...
];

%% Simulate
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t, x] = ode45(ode, tspan, xIC, opts);

%% Plot displacement
figure;
plot(t, x(:,1), 'LineWidth', 3,'Color','r');
grid on;
xlabel('Time (s)');
ylabel('Displacement x (m)');
title('Axial SDOF IC response (lumped beam model)');

%% Print key numbers (sanity check)
fprintf('K = %.3e N/m\n', K);
fprintf('M = %.3f kg\n', M);
fprintf('wn = %.2f rad/s (%.2f Hz)\n', wn, wn/(2*pi));
fprintf('zeta = %.4f (-)\n', zeta);
fprintf('B = %.3e N*s/m\n', B);
