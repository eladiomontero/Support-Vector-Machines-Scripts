load data/ripley

plot_options            = [];
plot_options.is_eig     = true;
plot_options.labels     = Y;
plot_options.title      = 'Ripley Dataset';
h1 = ml_plot_data(X,plot_options);
set(gca,'FontSize',14)
%% linear model
% note: X is train, Xt is test. They do not match at least, I checked

type='c'; 
gam = 1; 

[alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});
figure; plotlssvm({X,Y,type,gam,[],'lin_kernel','preprocess'},{alpha,b});
%error
[Yth, Zh] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);
err = sum(Yth~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)

% linear model is definetly not valid - 10.8% error, 108 missclass


%% RBF
% error - 9.4 % (94 missclass) - optimized from 
gam = 10;
sig2 = 1;
disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

% Plot the decision boundary of a 2-d LS-SVM classifier
plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

% Obtain the output of the trained classifier
[Yth, Zh] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
err = sum(Yth~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)

%% Using validation set - gridsearch

idx = randperm(size(X,1));

Xtrain = X(idx(1:200), :);
Ytrain = Y(idx(1:200));
Xval = X(idx(201:end), :);
Yval = Y(idx(201:end));

% 1 - calculate error on validation set, both gamma and and sig2
% trying different values
gamlist=[0.01,0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]; %13 values (cols)
sig2list=[0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100]; % 9 values (rows)

errlist = gridSearch(sig2list, gamlist, Xtrain, Ytrain, Xval, Yval, false);

% note: validation results are lower than test (makes sense - val set is smaller, overfit)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cross validation and finetuning   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Tunelssvm - it is shit,do not know why. ASK TA's
opt = 'csa';
opt_tune = 'gridsearch';
model = {X, Y, 'c', 10, 1, 'RBF_kernel', opt};
[gam, sigma, perf] = tunelssvm(model, opt_tune,...
    'crossvalidatelssvm', {10, 'misclass'});
%%
gam = 8.774;
sigma = 0.925;
[alpha,b] = trainlssvm({X,Y,type,gam,sigma,'RBF_kernel'});
figure; plotlssvm({X,Y,type,gam,sigma,'RBF_kernel','preprocess'},{alpha,b});
%error
[Yth, Zh] = simlssvm({X,Y,type,gam,sigma,'RBF_kernel'}, {alpha,b}, Xt);
err = sum(Yth~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)

% gam 10 and sigma 1 are top - 9.4% error.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              ROC curve            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % using test
[area,se,thresholds,FP,TP]=roc(Zh,Yt);
[thresholds FP TP];
% obtaining optimal bias
d = sqrt((1-TP).^2 + (0-FP).^2);
[~, i ] = min(d); % index of closest point to (1, 0)
biasOpt = thresholds(i);
set(gca,'FontSize',14)

%% correcting the bias
[alpha,b] = trainlssvm({X,Y,type,gam,sigma,'RBF_kernel'});
figure; plotlssvm({X,Y,type,gam,sigma,'RBF_kernel','preprocess'},{alpha, b - biasOpt});
%error
[Yth, Zh] = simlssvm({X,Y,type,gam,sigma,'RBF_kernel'}, {alpha, b - biasOpt}, Xt);
err = sum(Yth~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)

%% roc with val instead  of test

idx = randperm(size(X,1));

Xtrain = X(idx(1:200), :);
Ytrain = Y(idx(1:200));
Xval = X(idx(201:end), :);
Yval = Y(idx(201:end));

gam = 10;
sig2 = 1;

[alpha, b] = trainlssvm({Xtrain, Ytrain, 'c', gam, sig2});
[Ysim, Ylatent] = simlssvm({Xtrain, Ytrain, 'c', gam, sig2, ...
    'RBF_kernel'}, {alpha, b}, Xval);

[area,se,thresholds,FP,TP]=roc(Ylatent,Yval);
[thresholds FP TP];
% obtaining optimal bias
d = sqrt((1-TP).^2 + (0-FP).^2);
[~, i ] = min(d); % index of closest point to (1, 0)
biasOpt = thresholds(i);
set(gca,'FontSize',14)

