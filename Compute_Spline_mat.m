function [Spline_mat] = Compute_Spline_mat(X, numknots, precision)
    p = size(X, 2);
    out = 0;
    lastknot = p; 
    c_pt_times_all = [-2, 0:p/numknots:p, p+10]; 
    %disp(c_pt_times_all)
    % Define Tension Parameter
    ten_par = 0.5;  
    % Construct spline matrix
    values = 1:precision:lastknot;
    Spline_mat = zeros(length(values), numknots);
    for j=1:length(values)
        i = values(j);
        nearest_c_pt_index = max(find(c_pt_times_all<i));
        nearest_c_pt_time = c_pt_times_all(nearest_c_pt_index);
        next_c_pt_time = c_pt_times_all(nearest_c_pt_index+1);
        next2 = c_pt_times_all(nearest_c_pt_index+2);
        u = (i-nearest_c_pt_time)/(next_c_pt_time-nearest_c_pt_time);
        l = (next2-next_c_pt_time)/(next_c_pt_time-nearest_c_pt_time);
        r=[u^3 u^2 u 1]*[-ten_par 2-ten_par/l ten_par-2 ten_par/l;2*ten_par ten_par/l-3 3-2*ten_par -ten_par/l;-ten_par 0 ten_par 0;0 1 0 0];
        Spline_mat(j,nearest_c_pt_index-1:nearest_c_pt_index+2) = r;
    end
end
