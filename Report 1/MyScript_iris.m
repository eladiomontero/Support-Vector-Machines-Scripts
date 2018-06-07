load iris
%% Lin kernel
type='c'; 
gam = 10; 

[alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});
figure; plotlssvm({X,Y,type,gam,[],'lin_kernel','preprocess'},{alpha,b});
%error
[Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);
err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)


%% Poly kernel 
type='c'; 
gam = 1; 
t = 1; 
degree = 10;

[alpha,b] = trainlssvm({X,Y,type,gam,[t; degree],'poly_kernel'});
figure; plotlssvm({X,Y,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
%error
[Yht, Zt] = simlssvm({X,Y,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);
err = sum(Yht~=Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)
    

%% RBF kernel, sigma
disp('RBF kernel')
gam = 10; sig2list=[0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100];
errlist= gridSearch(sig2list, gam, X, Y, Xt, Yt, false)
% misclassif rate as function of sigma
figure;
plot(log10(sig2list), errlist, '*-'), 
xlabel('Log(sig2)'), ylabel('Number of misclass'), title('Gam = 10'),
set(gca,'FontSize',14)


%% RBF kernel, gam

sig2 = 1; gamlist=[0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10];
errlist = gridSearch(sig2, gamlist, X, Y, Xt, Yt, false)
%missclasif rate as function of gam
figure;
plot(log10(gamlist), errlist, '*-'), 
xlabel('Log(sig2)'), ylabel('Number of misclass'),, title("Sigma = 1"),
set(gca,'FontSize',14)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Hyperparameter tuning      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Using validation set - gridsearch
load iris
gam = 0.1;
sig2 = 20;

idx = randperm(size(X,1));

Xtrain = X(idx(1:80), :);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100), :);
Yval = Y(idx(81:100));

[alpha,b] = trainlssvm({Xtrain, Ytrain, 'c', gam, sig2, 'RBF_kernel'});
estYval = simlssvm({Xtrain, Ytrain,'c', gam, sig2, 'RBF_kernel'},{alpha,b},Xval);

err = sum(estYval~=Yval);
errlist=[errlist; err];

% 1 - calculate error on validation set, both gamma and and sig2
% trying different values
gamlist=[0.01,0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]; %13 values (cols)
sig2list=[0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100]; % 9 values (rows)

errlist = gridSearch(sig2list, gamlist, Xtrain, Ytrain, Xval, Yval, false).*100;

%% 3D plot of missclasifications
[X,Y] = meshgrid(log10(sig2list), log10(gamlist));
surf(X,Y,errlist'),xlabel('log(sigma)'), ylabel('log(gamma)'), zlabel('Number of misclass')
colormap summer
colorbar

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cross validation and finetuning   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% crossvalidate
[performance] = crossvalidate({X, Y, 'c', 100, 1, 'RBF_kernel'}, ... 
    10, 'misclass')

[performance] = leaveoneout({X, Y, 'c', 100, 1, 'RBF_kernel'}, ... 
    'misclass', 'mean')
% note: function returns the missclass in %, so to get the absolute number
% I multiply by the fold size (100/k=100/10=10)

%% Tunelssvm
opt = 'csa'
opt_tune = 'gridsearch'
for i=1:4
model = {X, Y, 'c', [], [], 'RBF_kernel', opt};
[gam, sig2] = tunelssvm(model, opt_tune,...
    'crossvalidatelssvm', {10, 'misclass'});
disp('XXXXXXXXXX');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              ROC curve            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gam = 0.01;
sig2 = 100;
idx = randperm(size(X,1));

Xtrain = X(idx(1:80), :);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100), :);
Yval = Y(idx(81:100));


[alpha, b] = trainlssvm({Xtrain, Ytrain, 'c', gam, sig2});
[Ysim, Ylatent] = simlssvm({Xtrain, Ytrain, 'c', gam, sig2, ...
    'RBF_kernel'}, {alpha, b}, Xval);
[area, se, deltab, TPR, FPR, ~, ~, ~, ~] = roc(Ylatent, Yval)
set(gca,'FontSize',14)


