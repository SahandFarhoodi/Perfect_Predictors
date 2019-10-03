function [statistics, ks_band, F, Z] = KS_statistics(spiketrain, X, b, p)
    
    spiketimes = find(spiketrain);

    one_X = [ones(size(X,1), 1), X];
    
    spiketimes = spiketimes(spiketimes>p); %Looking at spiketimes after first p miliseconds
    lambda = exp(one_X*b);
    Z = sum(lambda(1:spiketimes(1) - p));
    for i=1:size(spiketimes)-1
        Z = [Z, sum(lambda(spiketimes(i)-p+1:spiketimes(i+1)-p))];
    end
    Z = sort(Z);
    F = ecdf(Z);
    F = F';
    ks_band = 1.36/sqrt(size(Z,2));
    statistics = max(abs(F(2:end) - expcdf(Z,1)));
    
    
end

   