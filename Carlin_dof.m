function [out] = Carlin_dof(Y, X, b, N, rep, bin, Sigma, perf_thr, model)
    % Y: response, X: design matrix, b: coefficients vector
    % N: number of observations used to compute expectations
    % rep: number of times the Carlin_dof is computed
    % bin: the range of noise that we add to b1
    % model: the model under examination, Sigma: used if model is bayesian
    % perf_thr: the threshold for identifying perfect predictors
    [out] = zeros(1, rep);
    if (model == "usual" || model == "rem" || model == "spline")
        for i=1:rep
            disp(i);
            rnd = rand(size(b,1), N)*bin - bin/2;
            b_sample = repmat(b, 1, N) + rnd;
            lambda = exp(X*b_sample);
            Y_dup = repmat(Y, 1, N);
            f = -lambda + Y_dup.*log(lambda);
            logfYbeta = sum(f); %these are loglikelihood of beta given Y
            logfYbeta = logfYbeta + abs(min(logfYbeta));
            fYbeta = exp(logfYbeta)/sum(exp(logfYbeta)); %% this value is the probability of beta|Y
            M = repmat(fYbeta, size(b, 1), 1);
            Exp_b = (sum((b_sample.*M)'))'; %% Expectation of beta based on pdf beta|Y
            Dev = Deviance(Y, X, b_sample);
            Exp_of_Dev = sum(Dev.*fYbeta');
            Dev_of_Exp = Deviance(Y, X, Exp_b); 
            out(i) = Exp_of_Dev - Dev_of_Exp;
        end
    end
    if (model == "bayes")
        for i=1:rep
            rnd = rand(size(b,1), N)*bin - bin/2;
            b_sample = repmat(b, 1, N) + rnd;
            lambda = exp(one_X*b_sample);
            Y_dup = repmat(Y, 1, N);
            f = -lambda + Y_dup.*log(lambda);
            logfYbeta = sum(f); %these are log of f
            fYbeta = exp(logfYbeta)/sum(exp(logfYbeta));
            f = mvnpdf(b_sample', zeros(size(b_sample')), Sigma); %these are prior pdfs
            fYbeta = fYbeta.*(f)';
            fYbeta = fYbeta/sum(fYbeta);
            M = repmat(fYbeta, size(b, 1), 1);
            Exp_b = (sum((b_sample.*M)'))';
            Dev = Deviance(Y, one_X, b_sample);
            Exp_of_Dev = sum(Dev.*fYbeta');
            Dev_of_Exp = Deviance(Y, one_X, Exp_b); 
            out(i) = Exp_of_Dev - Dev_of_Exp;
        end
    end
    if (model == "ridge")
        for i=1:rep
            fprintf("%d\n", i);
            rnd = rand(size(b,1), N)*bin - bin/2;
            b_sample = repmat(b, 1, N) + rnd;
            lambda = exp(one_X*b_sample);
            Y_dup = repmat(Y, 1, N);
            f = -lambda + Y_dup.*log(lambda);
            logfYbeta = sum(f); %these are log of f
            fYbeta = exp(logfYbeta)/sum(exp(logfYbeta));
            f = zeros(size(fYbeta));
            for j = 1:size(b_sample,2)
                fprintf("%d ", j);
                W = diag(lambda(:, j));
                f(j) = det(one_X'*W*one_X)^(1/2);
            end
            fprintf("\n");
            f = f/sum(f);
            fYbeta = fYbeta.*f;
            fYbeta = fYbeta/sum(fYbeta);
            M = repmat(fYbeta, size(b, 1), 1);
            Exp_b = (sum((b_sample.*M)'))';
            Dev = Deviance(Y, one_X, b_sample);
            Exp_of_Dev = sum(Dev.*fYbeta');
            Dev_of_Exp = Deviance(Y, one_X, Exp_b); 
            out(i) = Exp_of_Dev - Dev_of_Exp;
        end
    end
    %{
    if (model == "bounded_max")
        for i=1:rep
            b_pos_bound = find(b < perf_thr & b > 9/10*perf_thr);
            b_neg_bound = find(b > -perf_thr & b < -9/10*perf_thr);
            rnd = rand(size(b,1), N)*bin - bin/2;
            rnd(b_pos_bound, :) = (rnd(b_pos_bound, :) - bin/2)/2;
            rnd(b_neg_bound, :) = (rnd(b_pos_bound, :) + bin/2)/2;
            b_sample = repmat(b, 1, N) + rnd;
            lambda = exp(one_X*b_sample);
            Y_dup = repmat(Y, 1, N);
            f = -lambda + Y_dup.*log(lambda);
            logfYbeta = sum(f); %these are log of f
            fYbeta = exp(logfYbeta)/sum(exp(logfYbeta));
            f = zeros(size(fYbeta));
            ind = find(max(b_sample(2:end, :)) < perf_thr);
            f(ind) = 1;
            f = f/sum(f);
            fYbeta = fYbeta.*f;
            fYbeta = fYbeta/sum(fYbeta);
            M = repmat(fYbeta, size(b, 1), 1);
            Exp_b = (sum((b_sample.*M)'))';
            Dev = Deviance(Y, one_X, b_sample);
            Exp_of_Dev = sum(Dev.*fYbeta');
            Dev_of_Exp = Deviance(Y, one_X, Exp_b); 
            out(i) = Exp_of_Dev - Dev_of_Exp;
        end
    end
    if (model == "bounded_lasso")
        for i=1:rep
            rnd = rand(size(b,1), N)*bin - bin/2;
            b_sample = repmat(b, 1, N) + rnd;
            lambda = exp(one_X*b_sample);
            Y_dup = repmat(Y, 1, N);
            f = -lambda + Y_dup.*log(lambda);
            logfYbeta = sum(f); %these are log of f
            fYbeta = exp(logfYbeta)/sum(exp(logfYbeta));
            f = zeros(size(fYbeta));
            ind = find(sum(abs(b_sample(2:end, :))) < perf_thr*(length(b)-1));
            f(ind) = 1;
            f = f/sum(f);
            fYbeta = fYbeta.*f;
            fYbeta = fYbeta/sum(fYbeta);
            M = repmat(fYbeta, size(b, 1), 1);
            Exp_b = (sum((b_sample.*M)'))';
            Dev = Deviance(Y, one_X, b_sample);
            Exp_of_Dev = sum(Dev.*fYbeta');
            Dev_of_Exp = Deviance(Y, one_X, Exp_b); 
            out(i) = Exp_of_Dev - Dev_of_Exp;
        end
    end
    %}
    if (model == "bounded_SS")
        for i=1:rep
            rnd = rand(size(b,1), N)*bin - bin/2;
            b_sample = repmat(b, 1, N) + rnd;
            lambda = exp(one_X*b_sample);
            Y_dup = repmat(Y, 1, N);
            f = -lambda + Y_dup.*log(lambda);
            logfYbeta = sum(f); %these are log of f
            fYbeta = exp(logfYbeta)/sum(exp(logfYbeta));
            f = zeros(size(fYbeta));
            ind = find(sum(b_sample(2:end, :).^2) < perf_thr^2*(length(b)-1));
            f(ind) = 1;
            f = f/sum(f);
            fYbeta = fYbeta.*f;
            fYbeta = fYbeta/sum(fYbeta);
            M = repmat(fYbeta, size(b, 1), 1);
            Exp_b = (sum((b_sample.*M)'))';
            Dev = Deviance(Y, one_X, b_sample);
            Exp_of_Dev = sum(Dev.*fYbeta');
            Dev_of_Exp = Deviance(Y, one_X, Exp_b); 
            out(i) = Exp_of_Dev - Dev_of_Exp;
        end
    end
    
end

