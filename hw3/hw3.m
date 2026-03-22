% Define helper function to build the constraint matrix
function M = quintic_matrix(t_i, t_f)
    M = [t_i^5, t_i^4, t_i^3, t_i^2, t_i, 1;
         t_f^5, t_f^4, t_f^3, t_f^2, t_f, 1;
         5*t_i^4, 4*t_i^3, 3*t_i^2, 2*t_i, 1, 0;
         5*t_f^4, 4*t_f^3, 3*t_f^2, 2*t_f, 1, 0;
         20*t_i^3, 12*t_i^2, 6*t_i, 2, 0, 0;
         20*t_f^3, 12*t_f^2, 6*t_f, 2, 0, 0];
end

% Solve for each curve segment
c1 = quintic_matrix(0, 1) \ [0; 2; 0; 1; 0; 1]
c2 = quintic_matrix(1, 3) \ [2; 4; 1; 1; 1; 1]
c3 = quintic_matrix(3, 5) \ [4; 5; 1; 0; 1; 0]

% Display coefficients
disp('Coefficients for curve 1 (t in [0,1]):'); disp(c1');
disp('Coefficients for curve 2 (t in [1,3]):'); disp(c2');
disp('Coefficients for curve 3 (t in [3,5]):'); disp(c3');

% Generate time vectors
t1 = linspace(0, 1, 100);
t2 = linspace(1, 3, 200);
t3 = linspace(3, 5, 200);

% Evaluate polynomials correctly (no flip!)
p1 = polyval(c1', t1);
p2 = polyval(c2', t2);
p3 = polyval(c3', t3);

% Plot
figure;
plot(t1, p1, 'b-', 'LineWidth', 2); hold on;
plot(t2, p2, 'r-', 'LineWidth', 2);
plot(t3, p3, 'g-', 'LineWidth', 2);

% Mark boundary points
plot([0,1,3,5], [0,2,4,5], 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

xlabel('Time t');
ylabel('Position p(t)');
title('Quintic Trajectory Segments');
legend('Segment 1', 'Segment 2', 'Segment 3', 'Boundary points');
grid on;
hold off;