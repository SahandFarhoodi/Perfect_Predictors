function [b6, cov_mod] = fit_mod(Y, X)
    one_X = [ones(size(X,1), 1), X];
    n=100;
    %precision of the algorithm
    delta = 0.00001;
    b6 = ones(size(one_X,2),1);
    new_b6 = b6;
    logbeta = [];
    thrd = 0.01;
    %The computations are based on the paper in the firth's method
    for i = (1:n)
        b6 = new_b6;
        etha = one_X*b6;
        mu = exp(etha);
        W = diag(mu);
        U = one_X'*(Y - mu);
        I = one_X'*W*one_X;
        new_b6 = b6 + inv(I)*U;
        Uabs = abs(U);
        ind = find(Uabs < thrd);
        new_b6(ind) = b6(ind);
        logbeta = [logbeta, new_b6];
        if (i > 1 & max(abs(logbeta(:,i) - logbeta(:,i-1)))  < delta)
            break;
        end
    end
    cov_mod = inv(I);
end