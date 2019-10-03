%% CRCNS data
%Total lenght of experiment: 60 s
%Train data: 39 s
%test data: 21 s

train_time = 39;
cr = load('current.txt');
vol = load('voltage_allrep.txt');
vol = vol(:, 1:6);
cr = cr(1:390000);
cr_all = repmat(cr, size(vol, 2), 1);
vol_all = reshape(vol, size(vol, 1)*size(vol, 2), 1);

p = 200; %number of history variables

%Merging any 10 time bins to get bins of length 1 milisecond for Voltage
mean_wind = 20;
vol_merge_integ = [];
vol_merge_max = [];
for s = 1: size(vol,2)
    v = vol_all((s-1)*size(vol, 1)+1: s*size(vol,1));
    v2 = v;
    v = max(reshape(v, mean_wind, length(v)/mean_wind))';
    v2 = sum(reshape(v2, mean_wind, length(v2)/mean_wind))';
    vol_merge_integ = [vol_merge_integ; v2];
    vol_merge_max = [vol_merge_max; v];
end
spiketrain = zeros(size(vol_merge_max, 1), 1);
ind = find(vol_merge_max > 20); 
spiketrain(ind, 1) = 1;
sum(spiketrain)/length(spiketrain)

% Finding v_rest
spiketimes = find(spiketrain);
v_rest = mean(vol_all(spiketimes-1));

%Merging any 10 time bins to get a bin of length 1 milisecond for Current
c = max(reshape(cr, mean_wind, length(cr)/mean_wind))';
curr_merge_max = repmat(c, size(vol, 2), 1);

%Smoothing current
mov_wind = 100;
curr_smooth = [];
for s = 1: size(vol,2)
    c = curr_merge_max((s-1)*size(vol, 1)/mean_wind+1 : s*size(vol, 1)/mean_wind);
    cnew = zeros(size(c));
    for i = 1 : length(c)
        cnew(i) = mean(c(max(1, i-mov_wind/2):min(length(c), i+mov_wind/2)));
    end
    curr_smooth = [curr_smooth; cnew];
end

% Seeing the relation between vol_merge_max and curr_smooth
%{
figure;
plot(vol_merge_max);
hold on;
plot(curr_smooth/30 - 65);
hold off;
legend('vol', 'smooth current');
%}

%Using smoothing splines to smooth I
%{
sm_cr = curr(1: length(curr)/size(vol, 2));
x_ax = (1:length(sm_cr))';
f = fit(x_ax, sm_cr,'smoothingspline', 'SmoothingParam', 0.01);
yy = feval(f, x_ax);
curr = repmat(yy, size(vol, 2), 1);
%}

%Constructing indicator functions of I and X_curr
Max_val = 900;
Min_val = -300;
ind = find(curr_smooth > Max_val);
curr_smooth(ind) = Max_val;
ind = find(curr_smooth <= Min_val);
curr_smooth(ind) = Min_val;
step = 100;
X_curr = zeros(size(spiketrain, 1), (Max_val - Min_val)/step - 1);
C = ceil((curr_smooth - Min_val)/step);
id0 = find(C==0);
C(id0) = 1;
for i = 2 : ceil(Max_val - Min_val)/step
    ind = find(C == i); 
    X_curr(ind, i-1) = 1;
end

% Removing rows excluded from history dependent model from X_curr
ind = [];
for s = 1: size(vol,2)
    ind = [ind, (s-1)*size(vol,1)/mean_wind+1: (s-1)*size(vol,1)/mean_wind + p];
end
ind = ind';
X_curr(ind, :) = [];
X_act_curr = curr_smooth;
X_act_curr(ind, :) = [];

% Constructing integrated vols
X_vol = [];
for s = 1: size(vol, 2)
    v = vol_merge_integ((s-1)*size(vol, 1)/mean_wind+1 : s*size(vol, 1)/mean_wind);
    sp = spiketrain((s-1)*size(vol, 1)/mean_wind+1 : s*size(vol, 1)/mean_wind);
    vol_new = [];
    sm = 0;
    for i = 1:length(v)
        vol_new = [vol_new; sm];
        if sp(i) == 1
            sm = 0;
        end
        if sp(i) ~= 1
            sm = sm - v(i) + v_rest;
        end
    end
    X_vol = [X_vol; vol_new(p+1: end)];
end
%{
plot(vol_new);
hold on;
plot(sp*40000);
hold off;
%}

% Constructing integrated Is
X_I = [];
for s = 1:size(vol, 2)
    Is = curr_smooth((s-1)*size(vol, 1)/mean_wind+1 : s*size(vol, 1)/mean_wind);
    sp = spiketrain((s-1)*size(vol, 1)/mean_wind+1 : s*size(vol, 1)/mean_wind);
    I_new = [];
    sm = 0;
    for i = 1:length(Is)
        I_new = [I_new; sm];
        if sp(i) == 1
            sm = 0;
        end
        if sp(i) ~= 1
            sm = sm + Is(i);
        end
    end
    X_I = [X_I; I_new(p+1: end)];
end
%{
plot(I_new);
hold on;
plot(sp*40000);
hold off;
%}

%Constructing history dependent part of the model
X_hist = [];
Y_all = [];
for s = 1: size(vol,2)
    disp(s);
    sp = spiketrain((s-1)*size(vol,1)/mean_wind+1: s*size(vol, 1)/mean_wind);
    %sp = find(sp);
    l = length(sp);
    Y_new = sp(p+1:end);
    X_new = [];
    for i = (1:p)
        X_new = [X_new, sp(p+1-i: l-i)];
    end
    X_hist = [X_hist; X_new];
    Y_all = [Y_all; Y_new];
end    

% Constructing the whole design matrix
%X_all = X_hist;
X_all = [X_hist, X_curr*20]; % 20 is for scale
%X_all = [X_hist, X_curr*20, X_vol, X_I]; % 20 is for scale
one_X_all = [ones(size(X_all,1),1), X_all];

%Choosing a subset of data to fit the model
st = 1;
fn = 60000; %just first experiment
X = X_all(st: fn, :);
one_X = [ones(size(X,1),1), X];
Y = Y_all(st: fn, :);
save('/projectnb/mpp-mog/sahand/Perfect_predictor/Report/X.mat', 'X');
save('/projectnb/mpp-mog/sahand/Perfect_predictor/Report/Y.mat', 'Y');

%% Usual glm
[b1, dev1, stats1] = glmfit(X, Y, 'poisson');
save('/projectnb/mpp-mog/sahand/Perfect_predictor/Report/b1.mat', 'b1');
%saveas(f, 'projectnb/mpp-mog/sahand/Perfect_predictors/Report', 'fig');

norm_ratio = normpdf(0)/normpdf(2); 
se_new = stats1.se;

perf_thr = -5;
%Removing ill columns and rows from the matrix
perfect_cols = find(b1(2:end) <= perf_thr);
ok_cols = find(b1(2:end)>perf_thr);
perfect_rows = [];
for i = (1:size(X,1))
    if (max(X(i,perfect_cols)) == 1)
        perfect_rows = [perfect_rows, i];
    end
end

for i = perfect_cols
    s = length(find(abs(X(:, i) > 0.001)));
    r = log(2/s);
    se_new(i+1) = r/2;
end

X_line = repmat((perfect_cols)', 2, 1);
Y_line = repmat([9;10], 1, length(perfect_cols));

%{
figure;
subplot(2,1,1);
plot(1:length(b1(2:end)), exp(b1(2:end)), 1:length(b1(2:end)), min(10, exp(b1(2:end)+2*stats1.se(2:end))), 1:length(b1(2:end)), min(10, exp(b1(2:end)-2*stats1.se(2:end))));
line(X_line, Y_line, 'color', 'green');
ylim([0,10]);
title({'CI''s obtained',  'from MLE'});
legend('Estimated value','Upper bound','Lower bound');
subplot(2,1,2);
plot(1:length(b1(2:end)), exp(b1(2:end)), 1:length(b1(2:end)), min(10, exp(b1(2:end)+2*se_new(2:end))), 1:length(b1(2:end)), min(10, exp(b1(2:end)-2*stats1.se(2:end))));
line(X_line, Y_line, 'color', 'green');
ylim([0,10]);
legend('Estimated value','Upper bound','Lower bound');
title({'CI''s computed by', 'normal approximation'});
%}

%% Removing GLM
X_rem = X;
X_rem(perfect_rows, :) = [];
X_rem(:, perfect_cols) = [];

Y_rem = Y;
Y_rem(perfect_rows) = [];
%Fitting a glm w.r.t. new matrix
[b2, dev2, stats2] = glmfit(X_rem, Y_rem, 'poisson');
one_X_rem = [ones(length(Y_rem), 1), X_rem];

%b2_full is the full vector of beta after adding Inf for perfect predictors
b2_full = b1;
b2_full(1) = max(b2(1), min(b1));
b2_full(ok_cols+1) = b2(2:end);
b2_full(perfect_cols+1) = min(b1); 

save('/projectnb/mpp-mog/sahand/Perfect_predictor/Report/b2.mat', 'b2');
save('/projectnb/mpp-mog/sahand/Perfect_predictor/Report/b2_full.mat', 'b2_full');
save('/projectnb/mpp-mog/sahand/Perfect_predictor/Report/X_rem.mat', 'X_rem');
save('/projectnb/mpp-mog/sahand/Perfect_predictor/Report/Y_rem.mat', 'Y_rem');

%% Spline transformations
%Finding the best number of knots based on aic:
[b4, dev4, stats4, Spline_mat] = fit_spline(Y, X, size(X, 2));
XSpline_mat = X*Spline_mat;
one_XSpline_mat = [ones(size(XSpline_mat,1),1),XSpline_mat];
save('/projectnb/mpp-mog/sahand/Perfect_predictor/Report/b4.mat', 'b4');
save('/projectnb/mpp-mog/sahand/Perfect_predictor/Report/b4.mat', 'b4');



%% Bounded Search
%n = number of iterations, delta = precision of the fitting process
%lambda = penalty constant
[b5_lasso, b5_max, b5_SS, cov_lasso, cov_max, cov_SS] = fit_bounded_search(Y, X, perf_thr);



