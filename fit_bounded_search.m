function [b5, cov, logbeta] = fit_bounded_search(Y, X, perf_thr)
    %one_X = [ones(size(X, 1), 1), X];
    n=100;
    delta = 0.001;
    b5 = ones(size(X,2),1);
    new_b5 = b5;
    logbeta = [];
    lambda = 1/20;
    for i = (1:n)
        %disp(i);
        b5 = new_b5;
        etha = X*b5;
        mu = exp(etha);
        W = sparse(length(mu), length(mu));
        W(1: length(mu)+1: end) = mu;
        %W = diag(mu);
        cov = inv(X'*W*X);
        logbeta = [logbeta, b5];
        if (i > 1 && max(abs(logbeta(:,i) - logbeta(:,i-1)))  < delta)
            break;
        end
        new_b5 = b5 + (X'*W*X)\(X'*(Y-mu));
        if (sum(new_b5.^2) > lambda*perf_thr^2*length(b5))
            break;
        end       
    end
end


