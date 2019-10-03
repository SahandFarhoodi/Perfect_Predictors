function out = KS_check(spiketrain, X, b, p)
    one_X = [ones(size(X,1), 1), X];
    spiketimes = find(spiketrain);
    spk = spiketimes(spiketimes>p); %Looking at spiketimes after first p miliseconds
    lambda = exp(one_X*b1);
    Z = sum(lambda(1:spk(1) - p));
    for i=1:size(spk)-1
        Z = [Z, sum(lambda(spiketimes(i)-p+1:spiketimes(i+1)-p))];
    end
    Z = sort(Z);
    F = ecdf(Z);
    out = (max(abs(F(2:end) - expcdf(Z,1))) > KS_band);
end