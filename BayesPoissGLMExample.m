b = [1;.1;-1000]; 
X = [ones(1000,1) normrnd(0,1,1000,1) binornd(1,.05,1000,1)];
Y = poissrnd(exp(X*b));
bhat = glmfit(X(:,2:end),Y,'poisson')

b0 = [0;0;0]; siginv = inv(1*eye(3));

beta(:,1) = [0;0;0];
for i = 1:10
    mu = exp(X*beta(:,i));
    SX = X.*sqrt(mu(:,ones(1,size(X,2))));
    %beta(:,i+1) = beta(:,i)+(SX'*SX+siginv)\(X'*(Y-mu)+siginv*b0); 
    beta(:,i+1) = beta(:,i)+(SX'*SX+siginv)\(X'*(Y-mu)-siginv*beta(:, i));
end
%beta

%%
% Script to generate and fit simulated data according to a conditional intensity
% with spline based history dependence
clear; rng(0);

% Define spline parameters
lastknot = 200; numknots = 5;
%c_pt_times_all = [-10 linspace(0,lastknot+1,numknots+1) lastknot+10];
c_pt_times_all = [-40 0 40 80 120 160 201 240];
s = 0.5;  % Define Tension Parameter

% Construct spline matrix
S = zeros(lastknot,numknots);
for i=1:lastknot
   nearest_c_pt_index = max(find(c_pt_times_all<i));
   nearest_c_pt_time = c_pt_times_all(nearest_c_pt_index);
   next_c_pt_time = c_pt_times_all(nearest_c_pt_index+1);
   next2 = c_pt_times_all(nearest_c_pt_index+2);
   u = (i-nearest_c_pt_time)/(next_c_pt_time-nearest_c_pt_time);
   l = (next2-next_c_pt_time)/(next_c_pt_time-nearest_c_pt_time);
   p=[u^3 u^2 u 1]*[-s 2-s/l s-2 s/l;2*s s/l-3 3-2*s -s/l;-s 0 s 0;0 1 0 0];
   S(i,nearest_c_pt_index-1:nearest_c_pt_index+2) = p;
end

% Simulate spiking activity
nsteps = 25000;
spiketrain = zeros(nsteps,1);
theta = [log(.05) 0 -10 -2 .1 .5 .1 0 0];
for i=lastknot+1:nsteps,
   lambda(i) = exp(theta*[1; S'*spiketrain(i-1:-1:i-lastknot)]);
   spiketrain(i) = min(poissrnd(lambda(i)),1);
end;

% Build design matrix for multiplicative history model
Hist = [];
for i=1:lastknot,
   Hist = [Hist spiketrain(lastknot-i+1:end-i)];
end;
X = Hist*S;
y=spiketrain(lastknot+1:end);

% Fit point process GLM
[b dev stats] = glmfit(X,y,'poisson');
[yhat,dylo,dyhi] = glmval(b,S,'log',stats);
ytrue = glmval(theta',S,'log'); %??????

% Plot results
plot(1:lastknot,ytrue,1:lastknot,yhat,1:lastknot,yhat+dyhi,'r--',1:lastknot,yhat-dylo,'r--');
xlabel('Lag (ms)');
ylabel('Intensity based on a single spike at given lag');
legend('True','Model fit','Error bounds');

% Now fit with Bayesian GLM
b0 = 0*ones(size(Hist,2)+1,1); 
sig = zeros(size(Hist,2)+1); sig(1,1)=.01; sig(2:end,2:end)=.01*toeplitz(.99.^[1:size(Hist,2)]); siginv = inv(sig);

X2 = [ones(size(Hist,1),1) Hist];

beta(:,1) = 0*ones(size(Hist,2)+1,1);
for i = 1:20
    mu = exp(X2*beta(:,i));
    SX = X2.*sqrt(mu(:,ones(1,size(X2,2))));
    %beta(:,i+1) = beta(:,i)+(SX'*SX+siginv)\(X2'*(y-mu)+siginv*b0); %works
    beta(:,i+1) = beta(:,i)+(SX'*SX+siginv)\(X2'*(y-mu)-siginv*beta(:, i)); %doesn't work
end
plot(1:lastknot,ytrue,1:lastknot,yhat,1:lastknot,exp(beta(1,end)+beta(2:end,end)));
legend('true','spline','bayes');

%figure; ind=find(y); plot(1:200,sum(Hist(ind,:)));

