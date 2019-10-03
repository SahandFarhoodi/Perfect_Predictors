function [b, dev, stats, Spline_mat_hist, Spline_mat_curr, best_numknots_hist, best_numknots_curr] = fit_spline(Y, X, hist_wind)
    %Finding the best number of knots based on aic:
    best_numknots_hist = 0;
    best_numknots_curr = 0;
    best_aic = inf;
    aic_log = [];
    %If I let p be really big then aic will be increasing
    for numknots_hist = 1:hist_wind/6+1
        for numknots_curr = 1:(size(X, 2)-hist_wind)/3+1
            Spline_mat_hist = Compute_Spline_mat(X(:, 1:hist_wind), numknots_hist, 1);
            Spline_mat_curr = Compute_Spline_mat(X(:, hist_wind+1:end), numknots_curr, 1);
            XSpline_mat_hist = X(:, 1:hist_wind)*Spline_mat_hist;
            XSpline_mat_curr = X(:, hist_wind+1: end)*Spline_mat_curr;
            XSpline_mat = [XSpline_mat_hist, XSpline_mat_curr];
            [b4, dev4, stats4] = glmfit(XSpline_mat, Y, 'poisson', 'constant', 'no');
            %Mu_spline = one_XSpline_mat*b4;
            Mu_spline = XSpline_mat*b4;
            l = sum(-exp(Mu_spline) + Y.*Mu_spline);
            aic = -2*l+2*length(b4);
            aic_log = [aic_log, aic];
            if (aic < best_aic)
                best_aic = aic;
                best_numknots_hist = numknots_hist;
                best_numknots_curr = numknots_curr;
            end
        end
    end
    
    %Now the best number of knots is found we fit it again:
    %best_numknots_curr = 4;  % This is set manually
    numknots_hist = best_numknots_hist;
    numknots_curr = best_numknots_curr;
    numknots_curr = 2;
    Spline_mat_hist = Compute_Spline_mat(X(:, 1:hist_wind), numknots_hist, 1);
    Spline_mat_curr = Compute_Spline_mat(X(:, hist_wind+1:end), numknots_curr, 1);
    XSpline_mat_hist = X(:, 1:hist_wind)*Spline_mat_hist;
    XSpline_mat_curr = X(:, hist_wind+1: end)*Spline_mat_curr;
    XSpline_mat = [XSpline_mat_hist, XSpline_mat_curr];    %one_XSpline_mat = [ones(size(XSpline_mat,1),1),XSpline_mat];
    [b, dev, stats] = glmfit(XSpline_mat, Y, 'poisson', 'constant', 'no');
end



