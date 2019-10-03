function [b8, cov_bayes, Sigma] = fit_bayes(Y, X, Sc_par, hist_wind)
    %one_X = [ones(size(X,1), 1), X];
    Sigma_up_left = give_Sigma(hist_wind, Sc_par);
    Sigma_low_right = give_Sigma(size(X, 2) - hist_wind, Sc_par/2);
    Sigma = zeros(size(X, 2));
    Sigma(1:hist_wind, 1:hist_wind) = Sigma_up_left;
    Sigma(hist_wind+1:end, hist_wind+1:end) = Sigma_low_right;
    
    n=100;
    delta = 0.001;
    b8 = zeros(size(X,2),1);
    new_b8 = b8;
    logbeta = b8;
    for i = (1:n)
        disp(i);
        b8 = new_b8;
        etha = X*b8;
        mu = exp(etha);
        W = sparse(length(mu), length(mu));
        W(1: length(mu)+1: end) = mu;
        new_b8 = b8 + (X'*W*X + inv(Sigma))\(X'*(Y-mu) - Sigma\b8);
        logbeta = [logbeta, new_b8];
        if (i > 1 && max(abs(logbeta(:,i) - logbeta(:,i-1)))  < delta)
            break;
        end
    end
    cov_bayes = inv(X'*W*X + inv(Sigma));
end