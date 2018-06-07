clear;
addpath('D:\Google Drive\KUL\Support Vector Machines\Report 1\LSSVM')
load('santafe.mat')

orders = 1:200;
lagmse = [];
Ztra = Z(1:800);
Zval = Z(801:1000);
nb = 200; 

for order=orders
    
    W = windowize(Ztra,1:order+1);
    X = W(:,1:order);
    Y = W(:,end);
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
    model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mse'});
    model = trainlssvm(model);
    Xs = Ztra(end-order+1:end,1);
    prediction=predict(model,Xs,nb);
    lagmse(length(lagmse)+1)=sum((prediction-Zval).^2)/length(Zval);
end
figure;
bar(categorical(orders), log(lagmse))
ylabel("MSE on validation set")
xlabel("Order")

inputs = bay_lssvmARD({X,Y,'f',531.9468,350.3872,'RBF_kernel'});



order = 50
W = windowize(Z,1:order+1);
X = W(:,1:order);
Y = W(:,end);
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mse'});
model = trainlssvm(model);
Xs = Z(end-order+1:end,1);
nb=200;
prediction=predict(model,Xs,nb);
sum((prediction-Ztest).^2)/length(prediction)
%0.05%predic
figure
hold on
plot(Ztest, 'r');
plot(prediction, 'k');
legend('Data Points', 'Prediction')


order = 108
W = windowize(Z,1:order+1);
X = W(:,1:order);
Y = W(:,end);
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm'; 
wFun = 'whuber'; 
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mse'});
model = trainlssvm(model);
Xs = Z(end-order+1:end,1);
nb=200;
prediction24=predict(model,Xs,nb);
sum((prediction24-Ztest).^2)/length(prediction24)
%0.05%predic
figure
hold on
plot(Ztest, 'k');
plot(prediction24, 'r');
legend('Data Points', 'Prediction')

order = 10
W = windowize(Z,1:order+1);
X = W(:,1:order);
Y = W(:,end);
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mse'});
model = trainlssvm(model);
Xs = Z(end-order+1:end,1);
nb=200;
prediction=predict(model,Xs,nb);
plot(prediction-Ztest);
legend('Ztest', 'Order=21', 'Order=10')

% 13% accuracy
sum((prediction-Ztest).^2)/length(prediction)
plot(prediction-Ztest)