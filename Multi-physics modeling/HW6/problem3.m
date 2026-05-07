%% Forced Convection Fin Heat Rejection — Motor Cooling Design
%
%  Additional Given Parameters (beyond Problem 2)
%  -----------------------------------------------------------------------
%  Subsystem      | Quantity                        | Symbol      | Value
%  -----------------------------------------------------------------------
%  Flow           | Free-stream velocity            | U_inf       | 10 m/s
%  Air at T_f     | Density                         | rho_fluid   | 1.127 kg/m^3
%  Air at T_f     | Dynamic viscosity               | mu_fluid    | 1.96e-5 Pa*s
%  Air at T_f     | Thermal conductivity            | k_fluid     | 0.028 W/(m*K)
%  Air at T_f     | Specific heat                   | c_p_fluid   | 1007 J/(kg*K)
%  -----------------------------------------------------------------------
%
%  All other parameters identical to Problem 2:
%  -----------------------------------------------------------------------
%  Heat load      | Motor input power               | P_motor     | 2000 W
%  Heat load      | Motor efficiency                | eta_motor   | 0.94
%  Heat load      | Required heat rejection         | q_required  | 120 W
%  Temperatures   | Base temperature                | T_b         | 80 °C
%  Temperatures   | Ambient air temperature         | T_inf       | 20 °C
%  Material       | Fin thermal conductivity (Al)   | k_fin       | 200 W/(m*K)
%  Geometry       | Motor outer diameter            | D           | 0.15 m
%  Geometry       | Motor axial length (= fin width)| w           | 0.15 m
%  Constraint     | Minimum fin spacing             | s_min       | 0.008 m
%  Search         | Fin thickness range             | t_range     | 0.002:0.0005:0.015 m
%  Search         | Fin radial length range         | L_range     | 0.005:0.001:0.025 m
%  -----------------------------------------------------------------------
%
%  Forced Convection Equations (F1-F5, evaluated once before sweep)
%  -----------------------------------------------------------------------
%  F1  | Film temperature                          | T_f = (T_b + T_inf) / 2
%  F2  | Reynolds number                           | Re_w = (rho_fluid * U_inf * w) / mu_fluid
%  F3  | Prandtl number                            | Pr = (c_p_fluid * mu_fluid) / k_fluid
%  F4a | Nusselt (laminar, Re_w <= 5e5)            | Nu_w = 0.664 * Re_w^(1/2) * Pr^(1/3)
%  F4b | Nusselt (turbulent w/ laminar LE, Re_w>5e5)| Nu_w = (0.037*Re_w^0.8 - 871) * Pr^(1/3)
%  F5  | Forced convection coefficient             | h_forced = (Nu_w * k_fluid) / w
%  -----------------------------------------------------------------------
%
%  Fin Design Equations (E1-E11, with h = h_forced)
%  -----------------------------------------------------------------------
%  E1  | Motor circumference              | P_max = pi * D
%  E2  | Fin cross-section area           | A_c   = w * t
%  E3  | Fin perimeter                    | P     = 2 * (w + t)
%  E4  | Fin parameter                    | m     = sqrt((P / A_c) * (h / k_fin))
%  E5  | Optimality indicator             | mL    = m * L
%  E6  | Heat per fin                     | q_fin = (T_b - T_inf) * sqrt(P * A_c * h * k_fin) * tanh(mL)
%  E7  | Number of fins (nearest even int)| N_fins = 2 * ceil(ceil(q_required / q_fin) / 2)
%  E8  | Total heat rejected              | q_total = N_fins * q_fin
%  E9  | Perimeter usage fraction         | perimeterUse = (N_fins * t) / P_max
%  E10 | Fin-to-fin spacing               | Space = (P_max / N_fins) - t
%  E11 | Fin efficiency                   | eta_f = tanh(mL) / mL
%  -----------------------------------------------------------------------

%% Given Parameters (Problem 2, unchanged)
P_motor   = 2000;         % Motor input power                  [W]
eta_motor = 0.94;         % Motor efficiency                   [-]
q_required = (1 - eta_motor) * P_motor;  % Required heat rejection [W]
T_b       = 80;           % Base temperature                   [°C]
T_inf     = 20;           % Ambient air temperature            [°C]
k_fin     = 200;          % Fin thermal conductivity (Al)      [W/(m*K)]
D         = 0.15;         % Motor outer diameter               [m]
w         = 0.15;         % Motor axial length (= fin width)   [m]
s_min     = 0.008;        % Minimum fin spacing                [m]
t_range   = 0.002:0.0005:0.015;  % Fin thickness range         [m]
L_range   = 0.005:0.001:0.025;   % Fin radial length range     [m]

%% Additional Given Parameters (Forced Convection)
U_inf     = 10;           % Free-stream velocity               [m/s]
rho_fluid = 1.127;        % Air density at T_f                 [kg/m^3]
mu_fluid  = 1.96e-5;      % Air dynamic viscosity at T_f       [Pa*s]
k_fluid   = 0.028;        % Air thermal conductivity at T_f    [W/(m*K)]
c_p_fluid = 1007;         % Air specific heat at T_f           [J/(kg*K)]

%% Forced Convection Equations (F1-F5)

% F1 - Film temperature [°C]
T_f = (T_b + T_inf) / 2;

% F2 - Reynolds number [-]
Re_w = (rho_fluid * U_inf * w) / mu_fluid;

% F3 - Prandtl number [-]
Pr = (c_p_fluid * mu_fluid) / k_fluid;

% F4 - Nusselt number [-]
if Re_w <= 5e5
    Nu_w = 0.664 * Re_w^(1/2) * Pr^(1/3);
    flow_regime = 'laminar (Re_w <= 5e5)';
else
    Nu_w = (0.037 * Re_w^0.8 - 871) * Pr^(1/3);
    flow_regime = 'turbulent w/ laminar LE (Re_w > 5e5)';
end

% F5 - Forced convection coefficient [W/(m^2*K)]
h_forced = (Nu_w * k_fluid) / w;

% Set h = h_forced for the fin design sweep
h = h_forced;

%% Design Search Loop
% E1 - Motor circumference [m]
P_max = pi * D;

% Conversion factor
m2in = 39.3701;

for i = 1:length(t_range)
    for j = 1:length(L_range)

        t = t_range(i);
        L = L_range(j);

        % E2 - Fin cross-section area [m^2]
        A_c = w * t;

        % E3 - Fin perimeter [m]
        P = 2 * (w + t);

        % E4 - Fin parameter [1/m]
        m = sqrt((P / A_c) * (h / k_fin));

        % E5 - Optimality indicator [-]
        mL = m * L;

        % E6 - Heat per fin [W]
        q_fin = (T_b - T_inf) * sqrt(P * A_c * h * k_fin) * tanh(mL);

        % E7 - Number of fins (nearest even integer) [-]
        N_fins = 2 * ceil(ceil(q_required / q_fin) / 2);

        % E8 - Total heat rejected [W]
        q_total = N_fins * q_fin;

        % E9 - Perimeter usage fraction [-]
        perimeterUse = (N_fins * t) / P_max;

        % E10 - Fin-to-fin spacing [m]
        Space = (P_max / N_fins) - t;

        % E11 - Fin efficiency [-]
        eta_f = tanh(mL) / mL;

        % Store results
        results(i,j).t             = t;
        results(i,j).L             = L;
        results(i,j).mL            = mL;
        results(i,j).q_fin         = q_fin;
        results(i,j).N_fins        = N_fins;
        results(i,j).q_total       = q_total;
        results(i,j).perimeterUse  = perimeterUse;
        results(i,j).Space         = Space;
        results(i,j).eta_f         = eta_f;

    end
end

%% Feasible Design Filter
feasible = [];
for i = 1:length(t_range)
    for j = 1:length(L_range)
        r = results(i,j);
        if r.perimeterUse < 1.0 && r.Space >= s_min
            feasible = [feasible; r.t, r.L, r.mL, r.q_fin, r.N_fins, ...
                        r.q_total, r.perimeterUse, r.Space, r.eta_f];
        end
    end
end

%% Display Forced Convection Calculation
fprintf('\nForced Convection Coefficient Calculation\n');
fprintf('------------------------------------------\n');
fprintf('Film Temperature       T_f       = %.1f C\n',        T_f);
fprintf('Free Stream Velocity   U_inf     = %.2f m/s\n',      U_inf);
fprintf('Air Density            rho_fluid = %.4f kg/m^3\n',   rho_fluid);
fprintf('Dynamic Viscosity      mu_fluid  = %.4e Pa*s\n',     mu_fluid);
fprintf('Thermal Conductivity   k_fluid   = %.4f W/mK\n',     k_fluid);
fprintf('Specific Heat          cp_fluid  = %.1f J/kgK\n',    c_p_fluid);
fprintf('Reynolds Number        Re_w      = %.3e\n',          Re_w);
fprintf('Prandtl Number         Pr        = %.4f\n',          Pr);
fprintf('Flow Regime                      = %s\n',            flow_regime);
fprintf('Nusselt Number         Nu_w      = %.4f\n',          Nu_w);
fprintf('Forced Convection h    h_forced  = %.4f W/m^2K\n',   h_forced);

%% Display Problem Parameters
fprintf('\nProblem Parameters\n');
fprintf('------------------\n');
fprintf('Motor Power            P_motor     = %d W\n',            P_motor);
fprintf('Motor Efficiency       eta_motor   = %.2f\n',            eta_motor);
fprintf('Required Heat Removal  q_Required  = %.2f W  [= (1 - eta) * P_motor]\n', q_required);
fprintf('Base Temperature       T_b         = %.1f C\n',          T_b);
fprintf('Ambient Temperature    T_inf       = %.1f C\n',          T_inf);
fprintf('Temperature Difference T_b-T_inf   = %.1f C\n',          T_b - T_inf);
fprintf('Fin Conductivity       k_fin       = %.1f W/mK\n',       k_fin);
fprintf('Fin Width (motor axis) w           = %.4f m  (%.4f in)\n', w, w*m2in);
fprintf('Motor Diameter         D           = %.4f m  (%.4f in)\n', D, D*m2in);
fprintf('Motor Circumference    Pmax        = %.4f m  (%.4f in)\n', P_max, P_max*m2in);
fprintf('Min Fin Spacing        smin        = %.4f m  (%.4f in)\n', s_min, s_min*m2in);
fprintf('Thickness Range        t_range     = [%.4f : %.4f] m\n',  t_range(1), t_range(end));
fprintf('Length Range           L_range     = [%.4f : %.4f] m\n',  L_range(1), L_range(end));
fprintf('Convection Coeff       h_forced    = %.4f W/m^2K\n',     h_forced);
fprintf('Total Feasible Designs             = %d\n',               size(feasible,1));
fprintf('Maximum q_fin (feasible designs)   = %.4f W\n',           feasible(1,4));


%% Plot 1: Feasible Designs Only
figure;
scatter3(feasible(:,1)*1e3, feasible(:,2)*1e3, feasible(:,5), 50, feasible(:,4), 'filled');
xlabel('Fin Thickness t [mm]');
ylabel('Fin Radial Length L [mm]');
zlabel('Number of Fins N_{fins}');
c = colorbar;
c.Label.String = 'Heat per Fin q_{fin} [W]';
colormap(jet);
title('Feasible Designs: N_{fins} vs (t, L), colored by q_{fin}');
view(135, 30);
grid on;

%% Plot 2: Full Design Space (infeasible gray, feasible colored)
t_mat = zeros(length(t_range), length(L_range));
L_mat = zeros(length(t_range), length(L_range));
N_mat = zeros(length(t_range), length(L_range));
q_mat = zeros(length(t_range), length(L_range));
feas_forc = false(length(t_range), length(L_range));

for i = 1:length(t_range)
    for j = 1:length(L_range)
        r = results(i,j);
        t_mat(i,j) = r.t;
        L_mat(i,j) = r.L;
        N_mat(i,j) = r.N_fins;
        q_mat(i,j) = r.q_fin;
        feas_forc(i,j) = (r.perimeterUse < 1.0) && (r.Space >= s_min);
    end
end

infeas_mask = ~feas_forc;

figure;
scatter3(t_mat(infeas_mask)*1e3, L_mat(infeas_mask)*1e3, N_mat(infeas_mask), ...
         15, [0.6 0.6 0.6], 'o');
hold on;
scatter3(t_mat(feas_forc)*1e3, L_mat(feas_forc)*1e3, N_mat(feas_forc), ...
         50, q_mat(feas_forc), 'filled');
xlabel('Fin Thickness t [mm]');
ylabel('Fin Radial Length L [mm]');
zlabel('Number of Fins N_{fins}');
c = colorbar;
c.Label.String = 'Heat per Fin q_{fin} [W]';
colormap(jet);
title('Full Design Space: Feasible (colored) vs Infeasible (gray)');
view(135, 30);
grid on;
hold off;

%% Plot showing natural vs forced convection design space
% feas_nat  = feasibility mask from Problem 2 (h = 10)
% feas_forc = feasibility mask from Problem 3 (h = h_forced)
feas_nat = feas_mask;
both_infeas = ~feas_nat & ~feas_forc;
nat_only    = feas_nat;              % feasible under natural convection
new_forced  = feas_forc & ~feas_nat; % gained by forced convection

figure;
scatter3(t_mat(both_infeas)*1e3, L_mat(both_infeas)*1e3, N_mat(both_infeas), ...
         15, [0.6 0.6 0.6], 'o');
hold on;
scatter3(t_mat(nat_only)*1e3, L_mat(nat_only)*1e3, N_mat(nat_only), ...
         40, [0.2 0.4 0.8], 'filled');
scatter3(t_mat(new_forced)*1e3, L_mat(new_forced)*1e3, N_mat(new_forced), ...
         50, [0.9 0.2 0.2], 'filled');
xlabel('Fin Thickness t [mm]');
ylabel('Fin Radial Length L [mm]');
zlabel('Number of Fins N_{fins}');
legend('Infeasible', 'Feasible (natural)', 'Newly feasible (forced)', ...
       'Location', 'best');
title('Design Space: Natural vs Forced Convection');
view(135, 30);
grid on;
hold off;

%% Plot: Natural vs Forced Convection Feasibility Comparison (2D)
figure;

% Infeasible in both — gray
both_infeas = ~feas_nat & ~feas_forc;
scatter(t_mat(both_infeas)*1e3, L_mat(both_infeas)*1e3, ...
        15, [0.6 0.6 0.6], 'o');
hold on;

% Feasible under forced convection — colored by q_fin
scatter(t_mat(feas_forc)*1e3, L_mat(feas_forc)*1e3, ...
        50, q_mat(feas_forc), 'filled');

% Natural convection feasibility boundary
contour(t_range*1e3, L_range*1e3, double(feas_nat)', [0.5 0.5], ...
        'k-', 'LineWidth', 2);

% Forced convection feasibility boundary
contour(t_range*1e3, L_range*1e3, double(feas_forc)', [0.5 0.5], ...
        'r--', 'LineWidth', 2);

xlabel('Fin Thickness t [mm]');
ylabel('Fin Radial Length L [mm]');
c = colorbar;
c.Label.String = 'Heat per Fin q_{fin} [W]';
colormap(jet);
legend('Infeasible', 'Feasible (forced)', ...
       'Natural convection boundary', 'Forced convection boundary', ...
       'Location', 'best');
title('Design Space: Natural vs Forced Convection');
grid on;
hold off;


%% Plot: Natural vs Forced Convection Feasibility Comparison (2D, 3 views)
figure;

%--- Subplot 1: Thickness vs Length ---
subplot(1,3,1);
scatter(t_mat(both_infeas)*1e3, L_mat(both_infeas)*1e3, ...
        15, [0.6 0.6 0.6], 'o');
hold on;
scatter(t_mat(feas_forc)*1e3, L_mat(feas_forc)*1e3, ...
        50, q_mat(feas_forc), 'filled');
contour(t_range*1e3, L_range*1e3, double(feas_nat)', [0.5 0.5], ...
        'k-', 'LineWidth', 2);
contour(t_range*1e3, L_range*1e3, double(feas_forc)', [0.5 0.5], ...
        'r--', 'LineWidth', 2);
xlabel('Fin Thickness t [mm]');
ylabel('Fin Radial Length L [mm]');
title('t vs L');
grid on;
hold off;

%--- Subplot 2: Length vs N_fins ---
subplot(1,3,2);
scatter(L_mat(both_infeas)*1e3, N_mat(both_infeas), ...
        15, [0.6 0.6 0.6], 'o');
hold on;
scatter(L_mat(feas_forc)*1e3, N_mat(feas_forc), ...
        50, q_mat(feas_forc), 'filled');
xlabel('Fin Radial Length L [mm]');
ylabel('Number of Fins N_{fins}');
title('L vs N_{fins}');
grid on;
hold off;

%% Plot: Natural vs Forced Convection Feasibility Comparison (2D, 3 views)
figure;

%--- Subplot 1: Thickness vs Length ---
subplot(1,3,1);
scatter(t_mat(both_infeas)*1e3, L_mat(both_infeas)*1e3, ...
        15, [0.6 0.6 0.6], 'o');
hold on;
scatter(t_mat(feas_forc)*1e3, L_mat(feas_forc)*1e3, ...
        50, q_mat(feas_forc), 'filled');
contour(t_range*1e3, L_range*1e3, double(feas_nat)', [0.5 0.5], ...
        'k-', 'LineWidth', 2);
contour(t_range*1e3, L_range*1e3, double(feas_forc)', [0.5 0.5], ...
        'r--', 'LineWidth', 2);
xlabel('Fin Thickness t [mm]');
ylabel('Fin Radial Length L [mm]');
title('t vs L');
grid on;
hold off;

%--- Subplot 2: Length vs N_fins ---
subplot(1,3,2);
scatter(L_mat(both_infeas)*1e3, N_mat(both_infeas), ...
        15, [0.6 0.6 0.6], 'o');
hold on;
scatter(L_mat(feas_forc)*1e3, N_mat(feas_forc), ...
        50, q_mat(feas_forc), 'filled');
xlabel('Fin Radial Length L [mm]');
ylabel('Number of Fins N_{fins}');
title('N_{fins} vs Fin Length');
grid on;
hold off;

%--- Subplot 3: Thickness vs N_fins ---
subplot(1,3,3);
scatter(t_mat(both_infeas)*1e3, N_mat(both_infeas), ...
        15, [0.6 0.6 0.6], 'o');
hold on;
scatter(t_mat(feas_forc)*1e3, N_mat(feas_forc), ...
        50, q_mat(feas_forc), 'filled');
xlabel('Fin Thickness t [mm]');
ylabel('Number of Fins N_{fins}');
title('N_{fins} vs Fin Thickness');
grid on;
hold off;

% Shared colorbar
c = colorbar;
c.Label.String = 'Heat per Fin q_{fin} [W]';
colormap(jet);
% Shared colorbar
c = colorbar;
c.Label.String = 'Heat per Fin q_{fin} [W]';
colormap(jet);


%% Top 10 Designs and Best Design Summary
% Sort feasible designs by descending q_fin
feasible = sortrows(feasible, -4);

fprintf('\n========================================================================\n');
fprintf('  TOP 10 DESIGNS - Ranked by Descending q_fin\n');
fprintf('========================================================================\n');
fprintf('%-6s %-10s %-10s %-10s %-10s %-8s %-8s %-8s %-10s %-10s\n', ...
        'Rank', 't[m]', 't[in]', 'L[m]', 'L[in]', 'N_fins', '%Perim', 'eta_f%', 'Space[m]', 'q_fin[W]');
fprintf('------------------------------------------------------------------------------\n');

N_top = min(10, size(feasible,1));
for k = 1:N_top
    fprintf('%-6d %-10.4f %-10.4f %-10.4f %-10.4f %-8d %-8.2f %-8.2f %-10.4f %-10.4f\n', ...
            k, ...
            feasible(k,1), feasible(k,1)*m2in, ...
            feasible(k,2), feasible(k,2)*m2in, ...
            feasible(k,5), ...
            feasible(k,7)*100, ...
            feasible(k,9)*100, ...
            feasible(k,8), ...
            feasible(k,4));
end

fprintf('\n========================================================================\n');
fprintf('  BEST DESIGN SUMMARY - Rank 1 (Highest q_fin)\n');
fprintf('========================================================================\n');
b = feasible(1,:);
fprintf('Fin Thickness    t       = %.4f m   (%.4f in)\n',   b(1), b(1)*m2in);
fprintf('Fin Length       L       = %.4f m   (%.4f in)\n',   b(2), b(2)*m2in);
fprintf('Number of Fins   N_fins  = %d\n',                    b(5));
fprintf('Optimality       mL      = %.4f\n',                  b(3));
fprintf('Fin Efficiency   eta_f   = %.2f %%\n',               b(9)*100);
fprintf('Perimeter Used   %%Perim  = %.2f %%\n',              b(7)*100);
fprintf('Fin Spacing      Space   = %.4f m   (%.4f in)\n',   b(8), b(8)*m2in);
fprintf('Heat per Fin     q_fin   = %.4f W\n',                b(4));
fprintf('Total Heat       q_total = %.4f W\n',                b(6));