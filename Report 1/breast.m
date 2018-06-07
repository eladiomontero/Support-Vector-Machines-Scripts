load breast;

scatter(trainset(:,1),trainset(:,2), 10, labels_train)
scatter(trainset(:,end-1),trainset(:,end), 10, labels_train)
colormap winter;

%% LINEAR
type='c'; 
gam = 1;
L_fold = 10;

[gam,sig2] = tunelssvm({trainset,labels_train,type,[],[],'lin_kernel'},'gridsearch',...
'crossvalidatelssvm',{L_fold,'misclass'});

[alpha,b] = trainlssvm({trainset,labels_train,type,gam,[],'lin_kernel'});
figure; plotlssvm({trainset,labels_train,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[labels_testh, Zh] = simlssvm({trainset,labels_train,type,gam,[],'lin_kernel'}, {alpha,b}, testset);
err = sum(labels_testh~=labels_test); 
fprintf('\n on test with gamma %f: #misclass = %d, error rate = %.2f%%\n', gam, err, err/length(labels_test)*100)
Y_latent = latentlssvm({trainset,labels_train,type,gam,[],'lin_kernel'},{alpha,b},testset);
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,labels_test);



%% RBF Kernel

type='c'; 
gam = 10;
L_fold = 10;
sig2 = 1;

[gam,sig2] = tunelssvm({trainset,labels_train,type,[],[],'RBF_kernel'},'gridsearch',...
'crossvalidatelssvm',{L_fold,'misclass'});

[alpha,b] = trainlssvm({trainset,labels_train,type,gam,sig2,'RBF_kernel'});
figure; plotlssvm({trainset(:,[1,2]),labels_train,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

[labels_testh, Zh] = simlssvm({trainset,labels_train,type,gam,sig2,'RBF_kernel'}, {alpha,b}, testset);
err = sum(labels_testh~=labels_test); 
fprintf('\n on test with gamma %f: #misclass = %d, error rate = %.2f%%\n', gam, err, err/length(labels_test)*100)
Y_latent = latentlssvm({trainset,labels_train,type,gam,sig2,'RBF_kernel'},{alpha,b},testset);
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,labels_test);
