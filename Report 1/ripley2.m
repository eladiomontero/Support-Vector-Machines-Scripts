load ripley

%Scatterplot
scatter(X(:,1),X(:,2), 10, Y)
colormap winter;

%The data overlaps a lot.
%2 dimensions, it looks like it's more separable on the second column
%Doesn't seem lineary separable.
%% LINEAR MODEL
type='c'; 
gam = 1;

[gam,sig2] = tunelssvm({X,Y,type,[],[],'lin_kernel'},'gridsearch',...
'crossvalidatelssvm',{L_fold,'misclass'});

[alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});
figure; plotlssvm({X,Y,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yth, Zh] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);
err = sum(Yth~=Yt); 
fprintf('\n on test with gamma %f: #misclass = %d, error rate = %.2f%%\n', gam, err, err/length(Yt)*100)
Y_latent = latentlssvm({X,Y,type,gam,[],'lin_kernel'},{alpha,b},Xt);
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,Yt);
%The error is 10.8% and the number of missclasses is 108 out of 1000.
%It's better to try with the RBF kernel

%% RBF Kernel
type = 'classification';
L_fold = 10;
 [gam,sig2] = tunelssvm({X,Y,type,[],[],'RBF_kernel'},'simplex',...
'crossvalidatelssvm',{L_fold,'misclass'});

[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

Y_latent = latentlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b},Xt);
[Yth, Zh] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
err = sum(Yth~=Yt);
fprintf('\n on test with gamma %f: #misclass = %d, error rate = %.2f%%\n', gam, err, err/length(Yt)*100)
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,Yt);

% Good fit on the RBF kernel even when we use our optimized value of gamma
% The parameters gamma and sigma change, but the AUC = 92-96
% The linear kernel has a AUC of 95, which is not much of improvement with
% respect to the RBF kernel.
%%