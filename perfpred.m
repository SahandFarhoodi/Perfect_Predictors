%% Simulation Data
load('/home/sahand/Softwares/Sahand/bin/Matlab_Codes/Perfect Predictors/spiketrain.mat');
spiketrain = spiketrain(1:10000);
spiketimes = find(spiketrain);
ISI = diff(spiketimes);
bins = 1:2:220;
%hist(ISI, bins);

corr_size = 100;
correlation = xcorr(spiketrain - mean(spiketrain), corr_size, 'coeff');
%plot (-corr_size:corr_size, correlation);

p = 200; %number of history variables
l = length(spiketrain);
Y = spiketrain(p+1:end);

X = [];
for i = (1:p)
    X = [X, spiketrain(p+1-i: l-i)];
end
one_X = [ones(size(X,1),1), X];

%% Real data: Ratina
load('/home/sahand/BU/Fall 2016/Directed Study - Neuroscience/Point Processing Problem Set/Retinal_ISIs.txt');
train=[];
%Retriving spiketrain from ISIs:
for i = 1:length(Retinal_ISIs)
    train = [train; zeros(Retinal_ISIs(i), 1)];
    train = [train; 1];
end
%using only first 10000 spikes so I can do the computations.
spiketrain = train(1:5000); % this is training data (bad names!) 
test = train(5001:14000); % this is test data (bad names!)
spiketimes = find(spiketrain);
ISI = diff(spiketimes);
bins = 1:2:220;
%hist(ISI, bins);
corr_size = 100;
correlation = xcorr(spiketrain - mean(spiketrain), corr_size, 'coeff');
%plot (-corr_size:corr_size, correlation);

p = 200; %number of history variables
l = length(spiketrain);
Y = spiketrain(p+1:end);

X = [];
for i = (1:p)
    X = [X, spiketrain(p+1-i: l-i)];
end
one_X = [ones(size(X,1),1), X];

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

% Extracting spikes from the vol (don't letting multiple successive spikes)
vol_to_spike = zeros(size(vol_all));
sp = find(vol_all > 10);
for i = 1:length(sp)
    if (i == 1)
        vol_to_spike(sp(i)) = 1;
    end
    if (i > 1 && sp(i-1) ~= sp(i) - 1)
        vol_to_spike(sp(i)) = 1;
    end
end

p = 200; %number of history variables

%Merging any 10 time bins to get bins of length 1 milisecond for Voltage
mean_wind = 10;
last_mom = floor(size(vol, 1)/mean_wind)*mean_wind;
vol_merge_integ = [];
vol_merge_max = [];
vol_merge_spike = [];
for s = 1: size(vol,2)
    v_spike = vol_to_spike((s-1)*size(vol, 1)+1: s*size(vol,1));
    v_spike = v_spike(1:last_mom);
    v_spike = max(reshape(v_spike, mean_wind, length(v_spike)/mean_wind))';
    v = vol_all((s-1)*size(vol, 1)+1: s*size(vol,1));
    v = v(1:last_mom);
    v2 = v;
    v = max(reshape(v, mean_wind, length(v)/mean_wind))';
    v2 = sum(reshape(v2, mean_wind, length(v2)/mean_wind))';
    vol_merge_integ = [vol_merge_integ; v2];
    vol_merge_max = [vol_merge_max; v];
    vol_merge_spike = [vol_merge_spike; v_spike];
end

%spiketrain = zeros(size(vol_merge_integ, 1), 1);
%ind = find(vol_merge_integ > 40); 
%spiketrain(ind, 1) = 1;
%It actually makes more sense, but result in all perfect predictors
spiketrain = vol_merge_spike;
sum(spiketrain)/length(spiketrain)

% Finding v_rest : Corrections here is needed
spiketimes = find(spiketrain);
v_rest = mean(vol_all(spiketimes-1));

%Merging any 10 time bins to get a bin of length 1 milisecond for Current
c = max(reshape(cr(1:last_mom), mean_wind, length(cr(1:last_mom))/mean_wind))';
curr_merge_max = repmat(c, size(vol, 2), 1);
c = mean(reshape(cr(1:last_mom), mean_wind, length(cr(1:last_mom))/mean_wind))';
curr_merge_mean = repmat(c, size(vol, 2), 1);

%Smoothing current
mov_wind = 100;
curr_smooth = [];
for s = 1: size(vol,2)
    c = curr_merge_max((s-1)*last_mom/mean_wind+1 : s*last_mom/mean_wind);
    cnew = zeros(size(c));
    for i = 1 : length(c)
        cnew(i) = mean(c(max(1, i-mov_wind/2):min(length(c), i+mov_wind/2)));
    end
    curr_smooth = [curr_smooth; cnew];
end
%curr_smooth = curr_merge_max;

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
step = 200;
X_curr = zeros(size(spiketrain, 1), (Max_val - Min_val)/step - 1);
C = ceil((curr_smooth - Min_val)/step);
id0 = find(C==0);
C(id0) = 1;
for i = 1 : ceil(Max_val - Min_val)/step
    ind = find(C == i); 
    X_curr(ind, i) = 1;
end

% Removing rows excluded from history dependent model from X_curr
ind = [];
for s = 1: size(vol,2)
    ind = [ind, (s-1)*last_mom/mean_wind+1: (s-1)*last_mom/mean_wind + p];
end
ind = ind';
X_curr(ind, :) = [];
X_act_curr = curr_smooth;
X_act_curr(ind, :) = [];

% Constructing integrated vols
%{
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
%}

% Constructing integrated Is
%{
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
%}

%Constructing history dependent part of the model
X_hist = [];
Y_all = [];
for s = 1: size(vol,2)
    disp(s);
    sp = spiketrain((s-1)*last_mom/mean_wind+1: s*last_mom/mean_wind);
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
X_all = [X_hist, X_curr];
one_X_all = [ones(size(X_all,1),1), X_all];
hist_wind = 200;

%% saving and loading data
save('../Data/design_matrices', 'X_all', 'Y_all', 'hist_wind');
design_matrices = load('../Data/design_matrices', 'X_all', 'Y_all', 'hist_wind');
X_all = design_matrices.X_all;
Y_all = design_matrices.Y_all;
hist_wind = design_matrices.hist_wind;

%% usual glm 
% choosing a subset of data
st = 1;
fn = 39000; %just first two experiment 
X = X_all(st: fn, :);
one_X = [ones(size(X,1),1), X];
Y = Y_all(st: fn, :);
[b1, dev1, stats1] = glmfit(X, Y, 'poisson', 'constant', 'off');
cov_usual = stats1.covb;
%[b2, dev2, stats2] = fit_standard_irls(X, Y, b1);
KS_plot(X, Y, b1, 1, 1, 1);
perf_thr = -10;
perfect_cols = find(b1 <= perf_thr);
ok_cols = find(b1 > perf_thr);
perfect_rows = [];
for i = (1:size(X,1))
    if (max(X(i,perfect_cols)) == 1)
        perfect_rows = [perfect_rows, i];
    end
end

%% glm with forcing perfect parameters to -infty
X_rem = X;
X_rem(perfect_rows, :) = [];
X_rem(:, perfect_cols) = [];
one_X_rem = [ones(size(X_rem, 1), 1), X_rem];
Y_rem = Y;
Y_rem(perfect_rows) = [];
[b2, dev2, stats2] = glmfit(X_rem, Y_rem, 'poisson', 'constant', 'off'); 
cov_rem = stats2.covb;
perfect_val = perf_thr;
b2_full = b1;
b2_full(ok_cols) = b2;
b2_full(perfect_cols) = perfect_val;
f = KS_plot(X, Y, b2_full, 1, 1, 1);

%% regularization method
%[b3 cov_firth] = fit_firth(Y, X);
lambda = 0.1;
[b3, cov_ridge, logbeta] = fit_ridge(Y, X, lambda);
KS_plot(X, Y, b3, 1, 1, 1);
figure; plot(b1); hold on; plot(b3); hold off;
%% Spline transformations
[b4, dev4, stats4, Spline_mat_hist, Spline_mat_curr, knots_hist, knots_curr] = fit_spline(Y, X, hist_wind);
XSpline_mat_hist = X(:, 1:hist_wind)*Spline_mat_hist;
XSpline_mat_curr = X(:, hist_wind+1: end)*Spline_mat_curr;
XSpline_mat = [XSpline_mat_hist, XSpline_mat_curr];
b4_stand = [Spline_mat_hist*b4(1:size(Spline_mat_hist, 2)); Spline_mat_curr*b4(size(Spline_mat_hist, 2)+1:end)];
cov_spline = stats4.covb;
%Spline_mat_hist = Compute_Spline_mat(X(:, 1:hist_wind), knots_hist, 1);
%Spline_mat_curr = Compute_Spline_mat(X(:, hist_wind+1:end), knots_curr, 1);

%% Bounded Search
[b5_SS, cov_SS, logbeta] = fit_bounded_search(Y, X, perf_thr);
KS_plot(X, Y, b5_SS, 1, 1, 1);
figure; plot(b1); hold on; plot(b5_SS); hold off;

%% Bayesian GLM - Cov matrix constructed with scale
Sc_par = 0.90;
[b8, cov_bayes, Sigma] = fit_bayes(Y, X, Sc_par, hist_wind);
KS_plot(X, Y, b8, 2, 1, 1);
figure; plot(b1); hold on; plot(b8); hold off;

%% Saving and loading data for all models
save('../Data/beta', 'X', 'Y', 'st', 'fn', 'b1', 'b2', 'b2_full', 'b3', 'b4', 'b5_SS', 'b8', 'cov_SS', 'cov_ridge', 'cov_bayes', 'cov_usual', 'cov_rem', 'cov_spline', 'X_rem', 'Y_rem', 'Spline_mat_hist', 'Spline_mat_curr', 'XSpline_mat_hist', 'XSpline_mat_curr', 'XSpline_mat', 'knots_hist', 'knots_curr', 'b4_stand', 'perfect_rows', 'perfect_cols', 'perf_thr', 'perfect_val', 'stats1', 'stats2', 'stats4');
models = load('../Data/beta');
X = models.X; Y = models.Y; st = models.st; fn = models.fn;
b1 = models.b1; b2 = models.b2; b2_full = models.b2_full; 
b3 = models.b3; b4 = models.b4; b8 = models.b8; b5_SS = models.b5_SS;
cov_SS = models.cov_SS; cov_bayes = models.cov_bayes; cov_ridge = models.cov_ridge;
cov_usual = models.cov_usual; cov_rem = models.cov_rem; cov_spline = models.cov_spline;
X_rem = models.X_rem; Y_rem = models.Y_rem; Spline_mat_hist = models.Spline_mat_hist;
Spline_mat_curr = models.Spline_mat_curr; XSpline_mat_hist = models.XSpline_mat_hist;
XSpline_mat_curr = models.XSpline_mat_curr; XSpline_mat = models.XSpline_mat;
knots_hist = models.knots_hist; knots_curr = models.knots_curr; b4_stand = models.b4_stand;
perfect_cols = models.perfect_cols; perfect_rows = models.perfect_rows;
perfect_val = models.perfect_val; perf_thr = models.perf_thr;
stats1 = models.stats1; stats2 = models.stats2; stats4 = models.stats4;
%%
% Correlation between one predictor and other predictors

col = 49; %This column is a perfect column
pp = perfect_cols(perfect_cols ~= col);
ind = find(pp > col);
pp(ind) = pp(ind) - 1;
pp = pp(pp<101);
X_line = repmat(pp', 2, 1);
Y_line = repmat([0.01; 0.015], 1, length(pp));

figure;
subplot(3,2,1);
c = [1:col-1, col+1:length(b1)];
corr = corr_usual(col, c);
plot(corr(1:100));
hold on;
line(X_line, Y_line, 'color', 'red');
hold off;
legend('correlation', 'perfect predictors');
title({'Standard IRLS'});

subplot(3,2,3);
corr = corr_ridge(col, c);
plot(corr(1:100));
ylabel('correlation with 49-th predictor');
title({'Ridge GLM'});

subplot(3,2,5);
corr = corr_bayes(col, c);
plot(corr(1:100));
xlabel('lag');
title({'Bayesian GLM'});

col = 60; %This column is not a perfect column
pp = perfect_cols(perfect_cols ~= col);
ind = find(pp > col);
pp(ind) = pp(ind) - 1;
pp = pp(pp<101);
X_line = repmat(pp', 2, 1);
Y_line = repmat([0.165; 0.185], 1, length(pp));

subplot(3,2,2);
c = [1:col-1, col+1:length(b1)];
corr = corr_usual(col, c);
plot(corr(1:100));
hold on;
line(X_line, Y_line, 'color', 'red');
hold off;
title({'Standard IRLS'});

subplot(3,2,4);
corr = corr_ridge(col, c);
plot(corr(1:100));
ylabel('correlation with 60-th predictor');
title({'Ridge GLM'});

subplot(3,2,6);
corr = corr_bayes(col, c);
plot(corr(1:100));
xlabel('lag');
title({'Bayesian GLM'});

%%

%KS_plots
%{
% For having KS_ratio use: txt = 'ratio = ??'; text(0.05,0.8,txt);    
figure; 
KS_plot(X, Y, b1, 2, 3, 1);
title({'Standard IRLS'});
KS_plot(X, Y, b2_full, 2, 3, 2);
title({'GLM with', 'forcing to -inf'});
KS_plot(X, Y, b8, 2, 3, 3);
title({'Bayesian GLM'});
KS_plot(X, Y, b3, 2, 3, 4);
title({'Ridge GLM'});
KS_plot(XSpline_mat, Y, b4, 2, 3, 5);
title({'Spline', 'transformation'});
KS_plot(X, Y, b5_SS, 2, 3, 6);
title({'SS bounded', 'search'});
%}

% Cross Validated Deviance and SSE
%{
% Computing Deviances
X_test = X_all(fn+1: 2*fn-st+1, :);
Y_test = Y_all(fn+1: 2*fn-st+1, :);
dev_test_usual = Deviance(Y_test, X_test, b1);
dev_test_rem = Deviance(Y_test, X_test, b2_full);
dev_test_ridge = Deviance(Y_test, X_test, b3);
dev_test_bayes = Deviance(Y_test, X_test, b8);
dev_test_SS = Deviance(Y_test, X_test, b5_SS);
XSpline_mat_test = X_test*Spline_mat;
dev_test_spline = Deviance(Y_test, XSpline_mat_test, b4);
just_one = ones(size(X,1), 1);
dev_test_null = Deviance(Y_test, just_one, b_null);

% Computing Deviance Ratios
cross_dev_ratio_usual = (dev_test_null - dev_test_usual)/dev_test_null;
cross_dev_ratio_rem = (dev_test_null - dev_test_rem)/dev_test_null;
cross_dev_ratio_ridge = (dev_test_null - dev_test_ridge)/dev_test_null;
cross_dev_ratio_bayes = (dev_test_null - dev_test_bayes)/dev_test_null;
cross_dev_ratio_SS = (dev_test_null - dev_test_SS)/dev_test_null;
cross_dev_ratio_spline = (dev_test_null - dev_test_spline)/dev_test_null;

% Computing SEE's
    SSE_usual = sum((Y_test-exp(X_test*b1)).^2);
    SSE_rem = sum((Y_test-exp(X_test*b2_full)).^2);
    SSE_ridge = sum((Y_test-exp(XSpline_mat_test*b4)).^2);
    SSE_spline = sum((Y_test-exp(X_test*b1)).^2);
    SSE_SS = sum((Y_test-exp(X_test*b5_SS)).^2);
    SSE_bayes = sum((Y_test-exp(X_test*b8)).^2);
%}

%% Normal Ratio CI
N = 10;
hw_Y = zeros(N,1);
eps = 0.00001;
lambda = [0:eps:5];
Likelihood = exp(-N*lambda).*lambda.^(sum(hw_Y))/(prod(factorial(hw_Y)));
%plot(lambda, Likelihood, 'r');
cum_Likelihood = cumsum(Likelihood)/sum(Likelihood);
answer = max(find(cum_Likelihood<0.95));
lambda(answer)
-log(0.05)/N
fprintf('The Bayesian CI for lambda is (0, %f).\n', -log(0.05)/N);
fprintf('The Bayesian CI for beta is (-inf, %f).\n',log(-log(0.05)/N));
%The theoretical Bayesian answer is -log(0.05)/N. How? 
%Compute probability distribution for lambda given X = (0, 0, ..., 0).
%You have to normalize the Likelihood here. After that find t such that
%integral from 0 to t is equal to 95%. (Like what we do generally for
%finding CIs.
%Note that here I've assumed that the priori distribution is uniform.

%Check the ratio from the normal approximation
norm_ratio = normpdf(0)/normpdf(1.96);
fprintf('The Normal-ratio based CI for lambda is (0, %f).\n', -log(1/norm_ratio)/N);
fprintf('The Normal-ratio based CI for lambda is (-inf, %f).\n', log(-log(1/norm_ratio)/N));

%% Computing Carling dof for different methods
% Forget about it for now
N = 1000;
rep = 50;
bin = 0.05;

[out] = Carlin_dof(Y, X, b1, N, rep, bin, Sigma, abs(perf_thr), "usual");
usual_dof_chain = Carlin_dof(Y, X, b1, N, rep, bin, Sigma, abs(perf_thr), "usual");
rem_dof_chain = Carlin_dof(Y, X, b1, N, rep, bin, Sigma, abs(perf_thr), "rem");
spline_dof_chain = Carlin_dof(Y, XSpline_mat, b4, N, rep, bin, Sigma, abs(perf_thr), "spline");
bayes_dof_chain = Carlin_dof(Y, X, b8, N, rep, bin, Sigma, abs(perf_thr), "bayes");
bounded_lasso_dof_chain = Carlin_dof(Y, X, b5_lasso, N, rep, bin, Sigma, abs(perf_thr), "bounded_lasso");
bounded_max_dof_chain = Carlin_dof(Y, X, b5_max, N, rep, bin, Sigma, abs(perf_thr), "bounded_max");
bounded_SS_dof_chain = Carlin_dof(Y, X, b5_SS, N, rep, bin, Sigma, abs(perf_thr), "bounded_SS");
N = 100;
rep = 1;
bin = 1;
firth_dof_chain = Carlin_dof(Y, X, b3, N, rep, bin, Sigma, abs(perf_thr), "firth"); % Too slow
firth_dof = mean(firth_dof_chain)/mean(usual_dof_chain)

usual_dof = mean(usual_dof_chain)/mean(usual_dof_chain)
rem_dof = mean(rem_dof_chain)/mean(usual_dof_chain)
spline_dof = mean(spline_dof_chain)/mean(usual_dof_chain)
bayes_dof = mean(bayes_dof_chain)/mean(usual_dof_chain)
bounded_lasso_dof = mean(bounded_lasso_dof_chain)/mean(usual_dof_chain)
bounded_max_dof = mean(bounded_max_dof_chain)/mean(usual_dof_chain)
bounded_SS_dof = mean(bounded_SS_dof_chain)/mean(usual_dof_chain)

%% Run it all multiple times 
%Compute deviance for different runs

rep = 100;
b_usual = zeros(length(b1), rep);
b_rem = zeros(length(b1), rep);
b_ridge = zeros(length(b1), rep);
b_spline = zeros(length(b1), rep);
b_SS = zeros(length(b1), rep);
b_bayes = zeros(length(b1), rep);

statistics_usual = zeros(rep, 1); 
statistics_rem = zeros(rep, 1);
statistics_ridge = zeros(rep, 1); 
statistics_spline = zeros(rep, 1);
statistics_SS = zeros(rep, 1);
statistics_bayes = zeros(rep, 1);

ks_band_usual = zeros(rep, 1); 
ks_band_rem = zeros(rep, 1);
ks_band_ridge = zeros(rep, 1); 
ks_band_spline = zeros(rep, 1);
ks_band_SS = zeros(rep, 1);
ks_band_bayes = zeros(rep, 1);

log_dev_usual = zeros(rep, 1);
log_dev_rem = zeros(rep, 1);
log_dev_ridge = zeros(rep, 1);
log_dev_spline = zeros(rep, 1);
log_dev_SS = zeros(rep, 1);
log_dev_bayes = zeros(rep, 1);

for j=1:rep
    %constructing data
    len = fn - st;
    s = ceil(rand(1) * (length(Y_all) - len));
    f = s + fn-st;
    X_new = X_all(s:f, :);
    Y_new = Y_all(s:f); 
    % usual
    fprintf("usual ");
    [b1_new, d1, s1] = glmfit(X_new, Y_new, 'poisson', 'constant', 'off');
    b_usual(:, j) = b1_new;
    perf_thr = -10;
    perfect_cols_new = find(b1_new <= perf_thr);
    ok_cols_new = find(b1_new > perf_thr);
    perfect_rows_new = [];
    for i = (1:size(X_new,1))
        if (max(X_new(i, perfect_cols_new)) == 1)
            perfect_rows_new = [perfect_rows_new, i];
        end
    end
    
    % removing
    fprintf("rem ");
    X_rem_new = X_new;
    X_rem_new(perfect_rows_new, :) = [];
    X_rem_new(:, perfect_cols_new) = [];
    Y_rem_new = Y_new;
    Y_rem_new(perfect_rows_new) = [];
    [b2_new, d2, s2] = glmfit(X_rem_new, Y_rem_new, 'poisson', 'constant', 'off'); 
    perfect_val = perf_thr;
    b2_full_new = b1_new;
    b2_full_new(ok_cols_new) = b2_new;
    b2_full_new(perfect_cols_new) = perfect_val;          
    
    %Ridge
    fprintf("ridge ");
    lambda = 0.2;
    [b3_new, cov_ridge_new, logbeta] = fit_ridge(Y_new, X_new, lambda);
    b_ridge(:, j) = b3_new;
  
    %spline
    fprintf("spline ");
    [b4_new, dev4_new, stats4_new, Spline_mat_new] = fit_spline(Y_new, X_new, size(X_new,2));
    XSpline_mat_new = X_new*Spline_mat_new;
    disp(size(b4_new));
    b_spline(:, j) = Spline_mat_new*b4_new;
    
    %bounded search
    fprintf("BS ");
    [b5_SS_new, cov_SS_new, logbeta] = fit_bounded_search(Y_new, X_new, perf_thr);
    b_SS(:, j) = b5_SS_new;
    
    % Bayesian GLM
    fprintf("bayes ");
    Sc_par = 0.9;
    [b8_new, cov_bayes_new, Sigma_new] = fit_bayes(Y_new, X_new, Sc_par);
    b_bayes(:, j) = b8_new;
    
    %Computing deviances
    log_dev_usual(j) = Deviance(Y_new, X_new, b1_new);
    log_dev_rem(j) = Deviance(Y_new, X_new, b2_full_new);
    log_dev_ridge(j) = Deviance(Y_new, X_new, b3_new);
    log_dev_spline(j) = Deviance(Y_new, XSpline_mat_new, b4_new);
    log_dev_SS(j) = Deviance(Y_new, X_new, b5_SS_new);
    log_dev_bayes(j) = Deviance(Y_new, X_new, b8_new);
end

%% Saving and Loading mult_run data
save('../Data/mult_run', 'b_usual', 'b_rem', 'b_spline', 'b_bayes', 'b_SS', 'b_ridge', 'log_dev_usual', 'log_dev_rem', 'log_dev_spline', 'log_dev_bayes', 'log_dev_SS', 'log_dev_ridge');
bb = load('../Data/mult_run.mat');
b_usual = bb.b_usual; b_rem = bb.b_rem; b_ridge = bb.b_ridge;
b_spline = bb.b_spline; b_bayes = bb.b_bayes; b_SS = bb.b_SS;
log_dev_usual = bb.log_dev_usual; log_dev_rem = bb.log_dev_rem;
log_dev_spline = bb.log_dev_spline; log_dev_bayes = bb.log_dev_bayes;
log_dev_ridge = bb.log_dev_ridge; log_dev_SS = bb.log_dev_SS;

%% Analysis 

% Confidence bands for different methods - one experiment and bootstrap
left = 0.08; sm_hor = 0.05; l_hor = 0.07; sm_w = 0.10;
l_w = 0.25; down = 0.1; h = 0.3; ver = 0.18;

up_col = [158,202,225]/255;
low_col = up_col;
est_col = [49, 130, 189]/255;
se_new = stats1.se;
for i = perfect_cols
    s = length(find(X(:, i) == 1));
    r = log(2/s);
    se_new(i) = r/2;
end

X_line = repmat((perfect_cols)', 2, 1);
Y_line = repmat([9;10], 1, length(perfect_cols));
hist_wind = 200; mx = 10;

% Second row
% Bayesian GLM
figure;
subplot('position', [left, down, l_w, h]);
se = diag(cov_bayes).^(1/2);
plot(exp(b8(1:hist_wind)), 'color', est_col); hold on;
plot(exp(b8(1:hist_wind)+2*se(1:hist_wind)), 'color', up_col);
plot(exp(b8(1:hist_wind)-2*se(1:hist_wind)), 'color', low_col); hold off;
line([1, hist_wind+30], [1, 1], 'LineStyle', '--', 'color', [0.5, 0.5, 0.5]);
txt = '\beta = 0';
text(200,1.5,txt, 'FontSize',8);
xlim([0, 230]);
ylim([0, 6]);
xlabel('lag');
ylabel('exp(\beta)');

subplot('position', [left+l_w+sm_hor, down, sm_w, h]);
m = 10;
temp = zeros(m*(length(b1)-hist_wind-1)+1, length(b1)-hist_wind);
temp(1, 1) = 1;
temp(size(temp, 1), size(temp, 2)) = 1;
for i = 1:length(b1)-hist_wind-1
    temp(m*(i-1)+1, i) = 1;
    for j = 2:m
        temp(m*(i-1)+j, [i, i+1]) = [(m-j+1)/m, (j-1)/m];
    end
end

est = temp*b8(hist_wind+1:end);
new_corr = sqrt(diag(temp*cov_bayes(hist_wind+1:end, hist_wind+1:end)*temp'));
up = temp*b8(hist_wind+1:end) + 2*new_corr;
low = temp*b8(hist_wind+1:end) - 2*new_corr;
%up = temp*(b8(hist_wind+1:end) + 2*se(hist_wind+1:end));
%low = temp*(b8(hist_wind+1:end) - 2*se(hist_wind+1:end));
ind = 1:1/m:6;
plot(ind, exp(est), 'color', est_col); hold on;
plot(ind, exp(up), 'color', up_col); hold on;
plot(ind, exp(low), 'color', low_col); hold off;
xlim([0, 7]);
ylim([0, 0.03]);
xticks([0 1 2 3 4 5 6 7]);
xlabel('Current level');

y = left;
x = l_w+sm_hor+sm_w;
annotation('textbox', [y, down+h, x, 0.07], 'String', '\bf Bayesian GLM \rm', 'EdgeColor', 'none', 'HorizontalAlignment', 'center')

%Spline Transformation
subplot('position', [left+l_w+sm_hor+sm_w+l_hor, down, l_w, h]);
%up = Spline_mat_hist * (b4(1:size(Spline_mat_hist, 2)) + 2*stats4.se(1:size(Spline_mat_hist, 2)));
%low = Spline_mat_hist * (b4(1:size(Spline_mat_hist, 2)) - 2*stats4.se(1:size(Spline_mat_hist, 2)));
covv = stats4.covb(1: size(Spline_mat_hist, 2), 1: size(Spline_mat_hist, 2));
new_se = sqrt(diag(Spline_mat_hist * covv * Spline_mat_hist'));
up = Spline_mat_hist * b4(1:size(Spline_mat_hist, 2)) + 2*new_se;
low = Spline_mat_hist * b4(1:size(Spline_mat_hist, 2)) - 2*new_se;
plot(exp(b4_stand(1:hist_wind)), 'color', est_col); hold on;
plot(exp(up), 'color', up_col); hold on;
plot(exp(low), 'color', low_col); hold off;
line([1, hist_wind+30], [1, 1], 'LineStyle', '--', 'color', [0.5, 0.5, 0.5]);
txt = '\beta = 0';
text(200,1.5,txt, 'FontSize',8);
xlim([0, 230]);
ylim([0, 6]);
xlabel('lag');
ylabel('exp(\beta)');

b4_curr = b4(knots_hist+4:end);
se_curr = stats4.se(knots_hist+4:end);
cov_curr = stats4.covb(knots_hist+4:end, knots_hist+4:end);
prec = 0.05;
tmp = Compute_Spline_mat(X(:, hist_wind+1:end), knots_curr, prec);
est1 = Spline_mat_curr*b4_curr;
up1 = Spline_mat_curr*b4_curr + 2*new_se1;
low1 = Spline_mat_curr*b4_curr - 2*new_se1;
est2 = tmp*b4_curr;
up2 = tmp*b4_curr + 2*new_se2;
low2 = tmp*b4_curr - 2*new_se2;

subplot('position', [left+l_w+sm_hor+sm_w+l_hor+l_w+sm_hor, down, sm_w, h]);
ind = 1:prec:6;
plot(ind, exp(est2), 'color', est_col); hold on;
plot(ind, exp(up2), 'color', up_col); hold on;
plot(ind, exp(low2), 'color', low_col); hold off;
xlim([0, 7]);
ylim([0, 0.03]);

xticks([0 1 2 3 4 5 6 7]);
xlabel('Current level');
y = left+l_w+sm_hor+sm_w+l_hor;
x = l_w+sm_hor+sm_w;
annotation('textbox', [y, down+h, x, 0.07], 'String', '\bf Spline transformation \rm', 'EdgeColor', 'none', 'HorizontalAlignment', 'center')

% First row
% Standard IRLS
subplot('position', [left, down+h+ver, l_w, h]);
plot(exp(b1(1:hist_wind)), 'color', est_col); hold on;
plot(min(mx, exp(b1(1:hist_wind)+2*stats1.se(1:hist_wind))), 'color', up_col);
plot(exp(b1(1:hist_wind)-2*stats1.se(1:hist_wind)), 'color', low_col); hold off;
line([1, length(b1(1:hist_wind))+30], [1, 1], 'LineStyle', '--', 'color', [0.5, 0.5, 0.5]);
txt = '\beta = 0';
text(201,1.5,txt, 'FontSize',8);
xlim([0, 230]);
ylim([0, 6]);
xlabel('lag');
ylabel('exp(\beta)');

subplot('position', [left+l_w+sm_hor, down+h+ver, sm_w, h]);
marg = 0.2;
for i = 1:size(b1)-hist_wind
    ind = hist_wind + i;
    upper = min(exp(b1(ind) + stats1.se(ind)), 10);
    lower = exp(b1(ind) - stats1.se(ind));
    est = exp(b1(ind));
    A = [1, 2];
    plot(i, est, '.', 'color', est_col, 'MarkerSize', 10); hold on;
    line([i, i], [lower, upper], 'LineStyle', '-', 'color', up_col); hold on;
    line([i-marg, i+marg], [lower, lower], 'LineStyle', '-', 'color', up_col); hold on;
    line([i-marg, i+marg], [upper, upper], 'LineStyle', '-', 'color', up_col); hold on;
end
xlim([0, 7]);
ylim([0, 0.03]);
xticks([0 1 2 3 4 5 6 7]);
xlabel('Current level');
hold off;

y = left;
x = l_w+sm_hor+sm_w;
annotation('textbox', [y, down+2*h+ver, x, 0.07], 'String', '\bf Standard IRLS \rm', 'EdgeColor', 'none', 'HorizontalAlignment', 'center')

% Bootstrap Standard IRLS
subplot('position', [left+l_w+sm_hor+sm_w+l_hor, down+h+ver, l_w, h]);
b_usual_mean = exp(mean(b_usual(1:hist_wind, :), 2));
upper = exp(quantile(b_usual(1:hist_wind, :), 0.975, 2));
lower = exp(quantile(b_usual(1:hist_wind, :), 0.025, 2));
plot(b_usual_mean, 'color', est_col); hold on;
plot(upper, 'color', up_col);  
plot(lower, 'color', low_col); hold off;
line([1, length(b1)+30], [1, 1], 'LineStyle', '--', 'color', [0.5, 0.5, 0.5]);
txt = '\beta = 0';
text(201,1.5,txt, 'FontSize',8);
xlim([0, 230]);
ylim([0, 6]);
xlabel('lag');
ylabel('exp(\beta)');

subplot('position', [left+l_w+sm_hor+sm_w+l_hor+l_w+sm_hor, down+h+ver, sm_w, h]);
marg = 0.2;
for ind = 1:size(b1)-hist_wind
    upper = exp(quantile(b_usual(hist_wind+ind, :), 0.975));
    lower = exp(quantile(b_usual(hist_wind+ind, :), 0.025));
    est = exp(mean(b_usual(hist_wind+ind, :)));
    plot(ind, est, '.', 'color', est_col, 'MarkerSize', 10); hold on;
    line([ind, ind], [lower, upper], 'LineStyle', '-', 'color', up_col); hold on;
    line([ind-marg, ind+marg], [lower, lower], 'LineStyle', '-', 'color', up_col); hold on;
    line([ind-marg, ind+marg], [upper, upper], 'LineStyle', '-', 'color', up_col); hold on;
end
xlim([0, 7]);
ylim([0, 0.03]);
xticks([0 1 2 3 4 5 6 7]);
xlabel('Current level');

y = left+l_w+sm_hor+sm_w+l_hor;
x = l_w+sm_hor+sm_w;
annotation('textbox', [y, down+2*h+ver, x, 0.07], 'String', '\bf Bootstrap for Standard IRLS \rm', 'EdgeColor', 'none', 'HorizontalAlignment', 'center')

tmp = zeros(2, 1);
tmp(1) = plot(NaN,NaN, '.', 'color', est_col, 'MarkerSize', 10);
tmp(2) = plot(NaN,NaN, '.', 'color', up_col, 'MarkerSize', 10);
legend(tmp, 'Estimated value','Confidence band');
hold off;

%%
% Comparing deviance for multiple runs
figure;
hist(log_dev_spline - log_dev_usual, 20)
title({'Difference of deviance:', 'Spline transformation vs Standard IRLS'});
ylabel('Frequency')
xlabel('Difference between deviances')

% Mean and sd of (Dev_model - Dev_usual) / Dev_usual
mean_dev_impr_rem = mean(log_dev_rem./log_dev_usual);
mean_dev_impr_bayes = mean(log_dev_bayes./log_dev_usual);
mean_dev_impr_ridge = mean(log_dev_ridge./log_dev_usual);
mean_dev_impr_spline = mean(log_dev_spline./log_dev_usual);
mean_dev_impr_SS = mean(log_dev_SS./log_dev_usual);
sd_dev_impr_rem = sqrt(var(log_dev_rem./log_dev_usual));
sd_dev_impr_bayes = sqrt(var(log_dev_bayes./log_dev_usual));
sd_dev_impr_ridge = sqrt(var(log_dev_ridge./log_dev_usual));
sd_dev_impr_spline = sqrt(var(log_dev_spline./log_dev_usual));
sd_dev_impr_SS = sqrt(var(log_dev_SS./log_dev_usual));

% Finding Bootstrap Confidence band for standard IRLS

