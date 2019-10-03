function [b3, cov_ridge, logbeta] = fit_ridge(Y, X, lambda)
    n=100;
    delta = 0.001;
    b3 = zeros(size(X, 2),1);
    new_b3 = b3;
    logbeta = b3;
    for i = (1:n)
        %disp(i);
        b3 = new_b3;
        etha = X*b3;
        mu = exp(etha);
        W = sparse(length(mu), length(mu));
        W(1: length(mu)+1: end) = mu;
        new_b3 = b3 + ((1-lambda)*X'*W*X + lambda*2)\((1-lambda)*X'*(Y-mu) - lambda*2*b3);
        logbeta = [logbeta, new_b3];
        if (i > 1 && max(abs(logbeta(:,i) - logbeta(:,i-1)))  < delta)
            break;
        end
    end
    cov_ridge = inv(X'*W*X + 2);
end