%% Radial Fin Heat Rejection — Motor Cooling Design 
% (Passive cooling, natural convection)
%
%  Given Parameters
%  -----------------------------------------------------------------------
%  Subsystem      | Quantity                        | Symbol      | Value
%  -----------------------------------------------------------------------
%  Heat load      | Motor input power               | P_motor     | 2000 W
%  Heat load      | Motor efficiency                | eta_motor   | 0.94
%  Heat load      | Required heat rejection         | q_required  | 120 W
%  Temperatures   | Base temperature                | T_b         | 80 °C
%  Temperatures   | Ambient air temperature         | T_inf       | 20 °C
%  Heat transfer  | Natural convection coefficient  | h           | 10 W/(m^2*K)
%  Material       | Fin thermal conductivity (Al)   | k_fin       | 200 W/(m*K)
%  Geometry       | Motor outer diameter            | D           | 0.15 m
%  Geometry       | Motor axial length (= fin width)| w           | 0.15 m
%  Constraint     | Minimum fin spacing             | s_min       | 0.008 m
%  Search         | Fin thickness range             | t_range     | 0.002:0.0005:0.015 m
%  Search         | Fin radial length range         | L_range     | 0.005:0.001:0.025 m
%  -----------------------------------------------------------------------
%
%  Equations
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

%% Given Parameters
P_motor   = 2000;         % Motor input power                  [W]
eta_motor = 0.94;         % Motor efficiency                   [-]
q_required = (1 - eta_motor) * P_motor;  % Required heat rejection [W]
T_b       = 80;           % Base temperature                   [°C]
T_inf     = 20;           % Ambient air temperature            [°C]
h         = 10;           % Natural convection coefficient     [W/(m^2*K)]
k_fin     = 200;          % Fin thermal conductivity (Al)      [W/(m*K)]
D         = 0.15;         % Motor outer diameter               [m]
w         = 0.15;         % Motor axial length (= fin width)   [m]
s_min     = 0.008;        % Minimum fin spacing                [m]
t_range   = 0.002:0.0005:0.015;  % Fin thickness range         [m]
L_range   = 0.005:0.001:0.025;   % Fin radial length range     [m]

%% Design Search Loop
% E1 - Motor circumference [m]
P_max = pi * D;

% Sweep through thicknesses
for i = 1:length(t_range)
    % Sweep throuhg the lengths
    for j = 1:length(L_range)

        t = t_range(i);    % Current fin thickness [m]
        L = L_range(j);    % Current fin radial length [m]

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

        % Store results in a structured array
        results(i,j).t             = t; % thickness [m]
        results(i,j).L             = L; % length [m]
        results(i,j).mL            = mL; % fin parameter
        results(i,j).q_fin         = q_fin; % heat rejected per fin [W]
        results(i,j).N_fins        = N_fins; % number of fins
        results(i,j).q_total       = q_total; % total heat rejected [W]
        results(i,j).perimeterUse  = perimeterUse; % fraction of perimeter use
        results(i,j).Space         = Space; % fin-to-fin spacing
        results(i,j).eta_f         = eta_f; % fin efficiency

    end
end

%% Display Results

% Conversion factor
m2in = 39.3701;  % m to inches

% Collect feasible designs
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

% Sort feasible designs by descending q_fin
feasible = sortrows(feasible, -4);

fprintf('\nProblem Parameters\n');
fprintf('------------------\n');
fprintf('Motor Power            P_motor     = %d W\n',            P_motor);
fprintf('Motor Efficiency       eta_motor   = %.2f\n',            eta_motor);
fprintf('Required Heat Removal  q_Required  = %.2f W  [= (1 - eta) * P_motor]\n', q_required);
fprintf('Base Temperature       T_b         = %.1f C\n',          T_b);
fprintf('Ambient Temperature    T_inf       = %.1f C\n',          T_inf);
fprintf('Temperature Difference T_b-T_inf   = %.1f C\n',          T_b - T_inf);
fprintf('Convection Coeff       h           = %.2f W/m^2K\n',     h);
fprintf('Fin Conductivity       k_fin       = %.1f W/mK\n',       k_fin);
fprintf('Fin Width (motor axis) w           = %.4f m  (%.4f in)\n', w, w*m2in);
fprintf('Motor Diameter         D           = %.4f m  (%.4f in)\n', D, D*m2in);
fprintf('Motor Circumference    Pmax        = %.4f m  (%.4f in)\n', P_max, P_max*m2in);
fprintf('Min Fin Spacing        smin        = %.4f m  (%.4f in)\n', s_min, s_min*m2in);
fprintf('Thickness Range        t_range     = [%.4f : %.4f] m\n',  t_range(1), t_range(end));
fprintf('Length Range           L_range     = [%.4f : %.4f] m\n',  L_range(1), L_range(end));
fprintf('Total Feasible Designs             = %d\n',               size(feasible,1));
fprintf('Maximum q_fin (feasible designs)   = %.4f W\n',           feasible(1,4));

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
feas_mask = false(length(t_range), length(L_range));

for i = 1:length(t_range)
    for j = 1:length(L_range)
        r = results(i,j);
        t_mat(i,j) = r.t;
        L_mat(i,j) = r.L;
        N_mat(i,j) = r.N_fins;
        q_mat(i,j) = r.q_fin;
        feas_mask(i,j) = (r.perimeterUse < 1.0) && (r.Space >= s_min);
    end
end

infeas_mask = ~feas_mask;

figure;
scatter3(t_mat(infeas_mask)*1e3, L_mat(infeas_mask)*1e3, N_mat(infeas_mask), ...
         15, [0.6 0.6 0.6], 'o');
hold on;
scatter3(t_mat(feas_mask)*1e3, L_mat(feas_mask)*1e3, N_mat(feas_mask), ...
         50, q_mat(feas_mask), 'filled');
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