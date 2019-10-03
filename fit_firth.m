function [out1, out2] = fit_firth(Y, X)
    one_X_sparse = sparse([ones(size(X,1),1),X]);
    %number of iterations
    n=100;
    %precision of the algorithm
    delta = 0.005;
    b3 = ones(size(one_X_sparse,2),1);
    new_b3 = b3;
    logbeta = [];
    %The computations are based on the paper in the firth's method
    for i = (1:n)
        disp(i);
        b3 = new_b3;
        etha = one_X_sparse*b3;
        mu = sparse(exp(etha));
        W = sparse(length(mu), length(mu));
        W(1:length(mu)+1:end) = mu;
        root_W = W.^(1/2);
        disp(size(root_W*one_X_sparse/(one_X_sparse'*W*one_X_sparse)*one_X_sparse'*root_W));
        M = sparse(diag(root_W*one_X_sparse/(one_X_sparse'*W*one_X_sparse)*one_X_sparse'*root_W));
        H = sparse(diag(M));
        new_b3 = b3 + (one_X_sparse'*W*one_X_sparse)\(one_X_sparse'*(Y - mu + H*(1/2-mu))); 
        logbeta = [logbeta, new_b3];
        if (i > 1 && max(abs(logbeta(:,i) - logbeta(:,i-1)))  < delta)
            break;
        end
        %if (i>1)
        %    disp(i);
        %    disp(max(abs(logbeta(:,i) - logbeta(:,i-1))));
        %end
    end
    cov_firth = inv(one_X_sparse'*W*one_X_sparse);
    out1 = b3;
    out2 = cov_firth;
end