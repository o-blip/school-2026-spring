%% Piero Risi Mortola
% Assignment 2 - MECE 6397
%

%% Problem 1: ydot + 2y = u(t), step input magnitude 3, zero ICs
clear; clc; close all

% Parameters
a = 2;          % coefficient on y(t)
U = 3;          % step magnitude
y0 = 0;         % y(0)

% Steady-state value from ODE: 0 + a*y_ss = U  => y_ss = U/a
y_ss = U/a;

% 2% settling time (first-order, monotone): |y_ss - y(t)| <= 0.02*y_ss
% For y(t) = y_ss(1 - exp(-a t)), error = y_ss*exp(-a t)
% exp(-a t_s) = 0.02  => t_s = -(1/a)*ln(0.02)
t2 = -(1/a)*log(0.02);

% Simulation horizon
t_final = 1.5 * t2;


% Step input u(t)
u = @(t) U*(t >= 0);



% Numerical simulation (ODE45)
f = @(t,y) -a*y + u(t);
[t_num, y_num] = ode45(f, [0 t_final], y0);

% Plot
figure; grid on; hold on
plot(t_num, y_num, 'LineWidth', 2)
xlabel('t (s)'); ylabel('y(t)')
xline(1.96,'k--')
yline(0.98*3/2,'k--')
legend('ode45', 'predicted settling time','2% off y_s_s', 'Location', 'southeast')
title('Step Response of ydot + 2y = 3u(t),  y(0)=0')

% Quick check at t2 (should be within 2% of y_ss)
y_at_t2 = y_ss*(1 - exp(-a*t2));
fprintf('y_ss = %.6f\n', y_ss);
fprintf('t_2%% = %.6f s\n', t2);
fprintf('y(t_2%%) = %.6f (error = %.2f%% of y_ss)\n', y_at_t2, 100*abs(y_ss - y_at_t2)/y_ss);
fprintf('t_final = %.6f s\n', t_final);


%% 3-DOF mass-spring-damper simulation with unit-step force on M2
clear; clc; close all

% Parameters
M1 = 3; M2 = 3; M3 = 3;                 % kg
K1 = 5; K2 = 10;                         % N/m
D1 = 4; D2 = 4; D3 = 1; D4 = 1;          % Ns/m

% Matrices (x = [x1;x2;x3])
M = diag([M1 M2 M3]);

D = [ D1+D3,    0,   -D3;
        0,   D2+D4,  -D4;
      -D3,    -D4,  D3+D4 ];

K = [ K1+K2,  -K2,   0;
      -K2,     K2,   0;
        0,      0,   0 ];

% Force input: f(t) = 1*u(t) applied to DOF 2
F = @(t) [0; 1*(t>=0); 0];

% State-space: z = [x; xdot],  zdot = A z + B f
Z0 = zeros(6,1);                         % zero ICs
Minv = inv(M);

A = [ zeros(3), eye(3);
     -Minv*K,  -Minv*D ];

B = [ zeros(3,1);
      Minv*[0;1;0] ];

% Simulate
tspan = [0 20];
odefun = @(t,z) A*z + B*(t>=0);          % unit step

opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t,z] = ode45(odefun, tspan, Z0, opts);

x  = z(:,1:3);
xd = z(:,4:6);

% Plot displacements
figure; grid on; hold on
plot(t,x(:,1),'LineWidth',2)
plot(t,x(:,2),'LineWidth',2)
plot(t,x(:,3),'LineWidth',2)
xlabel('t (s)'); ylabel('Displacement (m)')
legend('x_1','x_2','x_3','Location','best')
title('3-DOF Response to Unit-Step Force on DOF 2')

% % (Optional) Plot velocities
% figure; grid on; hold on
% plot(t,xd(:,1),'LineWidth',2)
% plot(t,xd(:,2),'LineWidth',2)
% plot(t,xd(:,3),'LineWidth',2)
% xlabel('t (s)'); ylabel('Velocity (m/s)')
% legend('\dot{x}_1','\dot{x}_2','\dot{x}_3','Location','best')
% title('Velocities')
