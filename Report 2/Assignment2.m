%% SETTINGS
format bank ; 

%% EXERCISE 1:
% Make sure u are in the SVM toolbox directory. 
% Create datapoints in a kind of trending fashion with a few outliers
% on graph and check results for several settings to make some
% very
% nice
% plots
%uiregress

%% EXERCISE 2.2: GRID SEARCH ON SUM OF COSINES  
    %% MAKE SOME DATA
    X = (-10:0.1:10)';
    Y = cos(X) + cos(2*X) + 0.1.*randn(length(X),1);
    Xtrain = X(1:2:length(X));
    Ytrain = Y(1:2:length(Y));
    Xtest = X(2:2:length(X));
    Ytest = Y(2:2:length(Y));
    gam = 50:500:10000;
    sig2 = 0.0001:0.05:3;
    MSE_mat = zeros(size(gam,2), size(sig2,2));
    %% DO SOME GRID SEARCH
    for g = 1:size(gam,2)
        for s = 1:size(sig2,2)
            [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam(g),sig2(s),'RBF_kernel'});
            Ypred = simlssvm({Xtrain,Ytrain,'f',gam(g),sig2(s),'RBF_kernel'},{alpha,b},Xtest);
            MSE_mat(g,s) = sum((simlssvm({Xtrain,Ytrain,'f',gam(g),sig2(s),'RBF_kernel'},{alpha,b},Xtest) - Ytest).^2)/size(Ytest,1);
        end
    end

    surf(sig2,gam,MSE_mat)
    xlabel('Sigma^2')
    ylabel('Gamma')
    zlabel('MSE')
    %% GET BEST MODEL
    min_MSE_mat = min(MSE_mat(:));
    [gam_idx, sig2_idx] = find(MSE_mat==min_MSE_mat);
    gam_min = gam(gam_idx);
    sig2_min = sig2(sig2_idx);
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam_min,sig2_min,'RBF_kernel'});
    YtestEst = simlssvm({Xtrain,Ytrain,'f',gam_min,sig2_min,'RBF_kernel'},{alpha,b},Xtest);
    %% MAKE SOME PLOT S

    plot(Xtest,Ytest,'.');
    hold on;
    plot(Xtest,YtestEst);
    legend('Y test', 'Y predicted');
   
    %% TUNE SOME HYPERPARAMETERS
    %cost_crossval = crossvalidate({Xtrain,Ytrain,'f',gam,sig2},10);
    %cost_loo = leaveoneout({Xtrain,Ytrain,'f',gam,sig2});
    %% COMPARE SOME TESTS
    optFuns = {'simplex', 'gridsearch'};
    globalOptFuns = {'ds', 'csa'};
    res = {};
    no_repeats = 10;
    varNames = {'optFun', 'globalOptFun', 'Time', 'Rep', 'Gam', 'Sig2', 'MSE'};
    df = table;
    for optFun = optFuns
        for globalOptFun = globalOptFuns
            for rep = 1 : no_repeats
                tic;
                row = size(df,1)+1;
                [gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel', globalOptFun{1} },optFun{1} ,'crossvalidatelssvm',{10,'mse'})
                df(row,:) = table(optFun,globalOptFun,toc,rep,gam,sig2,cost);
            end  
        end
    end
    df.Properties.VariableNames = varNames;
    writetable(df, 'exc2.2.csv');

    %%
    optFuns = {'simplex', 'gridsearch'};
    globalOptFuns = {'ds', 'csa'};
    res = {};
    no_repeats = 10;
    varNames = {'optFun', 'globalOptFun', 'Time', 'Rep', 'Gam', 'Sig2', 'MSE'};
    for optFun = optFuns
        for globalOptFun = globalOptFuns
            for rep = 1 : no_repeats
                [gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel', 'csa' },'gridsearch','crossvalidatelssvm',{10,'mse'}) 
                disp(strcat(num2str(gam), num2str(sig2)));
            end
        end
    end
%% EXERCISE 2.3 BAYESIAN FRAMEWORK
    %% REGRESSION
    sig2 = 2; gam = 600;
    criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1);
    criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2);
    criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3);
    [~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
    [~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
    [~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);
    sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');
	%% CLASSIFICATION 
    clear;
    load iris;
    gam = [0.01 1 10 100];
    sig2 = 10;
    for g=gam
            bay_modoutClass({X,Y,'c',g,sig2},'figure');
    end
    % effect of gamma
    sig2 = 0.75;
    bay_modoutClass({X,Y,'c',0.001,sig2},'figure');
    bay_modoutClass({X,Y,'c',0.5,sig2},'figure');
    bay_modoutClass({X,Y,'c',5,sig2},'figure');
    bay_modoutClass({X,Y,'c',10,sig2},'figure');
    colorbar
    
    gam = 5;
    bay_modoutClass({X,Y,'c',gam,0.001},'figure');
    bay_modoutClass({X,Y,'c',gam,0.5},'figure');
    bay_modoutClass({X,Y,'c',gam,0.75},'figure');
    bay_modoutClass({X,Y,'c',gam,5},'figure');
    colorbar
    
    gam = 5; sig2 = 0.75;
     bay_modoutClass({X,Y,'c',gam,sig2},'figure'); 
    colorbar;
    %% AUTOMATIC RELEVANCE DETERMINATION
    type = 'function approximation';
    X = 10.*rand(100,3)-3; 
    Y = cos(X(:,1)) + cos(2*(X(:,1))) + 0.3.*randn(100,1); 
    [dimensions, ordered, costs, sig2s] = bay_lssvmARD({X,Y,'class',gam,sig2}); 
%% EXERCISE 2.5: ROBUST REGRESSION
    %% SOME DATA
    clear; clc;
    X = (-10:0.2:10)';
    Y = cos(X) + cos(2*X) + 0.1.*rand(size(X));
    
    %% ADD OUTLIERS
    out = [15 17 19 2 54 87 5 52 96 99 20];
    Y(out) = 0.7+0.3*rand(size(out));
    out = [41 44 46 65 98 78 2 58 69 25 87];
    Y(out) = 1.5+0.2*rand(size(out)); 
    gam = 100; sig2 = 0.1;
    [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
    plotlssvm({X,Y,'f',gam, sig2, 'RBF_kernel'}, {alpha,b})
    
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel'); 
    costFun = 'rcrossvalidatelssvm'; 
     wFun = 'whuber'; 
     model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun); 
      model = robustlssvm(model); 
      Ypred = simlssvm({X,Y,'f',model.gam,model.kernel_pars,'RBF_kernel'},{model.alpha,model.b},X);
      plotlssvm(model); 
    %% MSE
%     model=tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'mse'});
%     model=trainlssvm({X,Y,'f',model.gam(end),model.kernel_pars,'RBF_kernel'});
%     Ypred = simlssvm({X,Y,'f',model.gam,model.kernel_pars,'RBF_kernel'},{model.alpha,model.b},X);  
%     plot(X,Ypred);
%     
%     model = initlssvm(X,Y,'f',[],[],'RBF_kernel');

    %% WLOGISTIC
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
    model=tunelssvm(model,'simplex', 'rcrossvalidatelssvm',{10,'mae'},'wlogistic');
    model=robustlssvm(model);
    YpredLog = simlssvm({X,Y,'f',model.gam(end),model.kernel_pars,'RBF_kernel'},{model.alpha,model.b},X); 
    
    
    %% WHUBER
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
    model=tunelssvm(model,'simplex', 'rcrossvalidatelssvm',{10,'mae'},'whuber');
    model=robustlssvm(model);
    YpredHub = simlssvm({X,Y,'f',model.gam,model.kernel_pars,'RBF_kernel'},{model.alpha,model.b},X); 
    
    
    %% WHAMPEL
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
    model=tunelssvm(model,'simplex', 'rcrossvalidatelssvm',{10,'mae'},'whampel');
    model=robustlssvm(model);
    YpredHam = simlssvm({X,Y,'f',model.gam,model.kernel_pars,'RBF_kernel'},{model.alpha,model.b},X); 
    
    
    %% WMYRIAD
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
    model=tunelssvm(model,'simplex', 'rcrossvalidatelssvm',{10,'mae'},'wmyriad');
    model=robustlssvm(model);
    YpredMyr = simlssvm({X,Y,'f',model.gam,model.kernel_pars,'RBF_kernel'},{model.alpha,model.b},X); 
    
    figure
    plot(X, Y, '.', X, YpredLog, X, YpredHub, X, YpredMyr, X, YpredHam)
    
    legend("Y", "Logistic", "Huber", "Hampel", "Myriad") 
    
    

    
    
    
    


