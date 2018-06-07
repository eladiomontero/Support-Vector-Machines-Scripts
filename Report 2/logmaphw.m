clear;
addpath('D:\Google Drive\KUL\Support Vector Machines\Report 1\LSSVM')
load('logmap.mat')
orders = 1:50;
lagmse = [];
Ztra = Z(1:100);
Zval = Z(101:150);
lagmse = [];
for order=orders
    
    W = windowize(Ztra,1:order+1);
    X = W(:,1:order);
    Y = W(:,end);
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
    model = tunelssvm(model,'gridsearch','crossvalidatelssvm',{10,'mse'});
    model = trainlssvm(model);
    Xs = Ztra(end-order+1:end,1);
    nb=50;
    prediction24=predict(model,Xs,nb);
    lagmse(length(lagmse)+1)=sum((prediction24-Zval).^2)/length(Zval);
end

bar(categorical(orders), lagmse)
plot(lagmse)
ylabel("MSE")
xlabel("Order")
ylim([0 0.2])

order = 24
W = windowize(Z,1:order+1);
X = W(:,1:order);
Y = W(:,end);
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm'; 
wFun = 'whuber'; 
model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun); 
model = robustlssvm(model);
Xs = Z(end-order+1:end,1);
nb=50;
prediction24=predict(model,Xs,nb);
sum((prediction24-Ztest).^2)/length(prediction24)
%0.05%predic
figure
hold on
plot(Ztest, 'k');
plot(prediction24, 'r');
legend('Data Points', 'Prediction')

order = 11
W = windowize(Z,1:order+1);
X = W(:,1:order);
Y = W(:,end);
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mse'});
model = trainlssvm(model);
Xs = Z(end-order+1:end,1);
nb=50;
prediction11=predict(model,Xs,nb);
plot(prediction11);
legend('Ztest', 'Order=21', 'Order=10')

% 13% accuracy
sum((prediction11-Ztest).^2)/length(prediction11)
plot(prediction11-Ztest)