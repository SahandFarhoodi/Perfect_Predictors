function [b2, dev2, stats2] = fit_standard_irls(X, Y, b1)
    perf_thr = -10;
    perfect_cols = find(b1 <= perf_thr);
    ok_cols = find(b1 > perf_thr);
    perfect_rows = [];
    for i = (1:size(X,1))
        if (max(X(i,perfect_cols)) == 1)
            perfect_rows = [perfect_rows, i];
        end
    end

    X_rem = X;
    X_rem(perfect_rows, :) = [];
    X_rem(:, perfect_cols) = [];
    one_X_rem = [ones(size(X_rem, 1), 1), X_rem];
    Y_rem = Y;
    Y_rem(perfect_rows) = [];
    disp("here");
    [b2, dev2, stats2] = glmfit(X_rem, Y_rem, 'poisson', 'constant', 'off'); 
    disp("here2");
    perfect_val = perf_thr;
    b2_full = b1;
    b2_full(ok_cols) = b2;
    b2_full(perfect_cols) = perfect_val;
    b2 = b2_full;
end
