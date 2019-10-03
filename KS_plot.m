function out = KS_plot(X, Y, b, x, y, z)
    % Y = response vector
    % X = design matrix
    % b = coefficients of the model
    % x, y and z = information for subplot
    %one_X = [ones(size(X,1),1), X];
    one_X = X;
    sp = find(Y);
    lambda = exp(one_X*b);
    Z = sum(lambda(1:sp(1)));
    for i=1:size(sp)-1
        Z = [Z, sum(lambda(sp(i):sp(i+1)))];
    end
    [eCDF, zvals] = ecdf(Z);
    mCDF = 1 - exp(-zvals);
    %figure();
    subplot(x, y, z);
    f = plot(eCDF, mCDF); 
    hold on;
    KS_band = 1.36/sqrt(size(Z,2));
    plot([0 1], [KS_band 1+KS_band], 'red', [0 1], [-KS_band 1-KS_band], 'red');
    hold off;
    ylim([0,1]);
    out = 1;
end

