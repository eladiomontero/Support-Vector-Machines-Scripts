clear;
load digits; 
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X)); 
maxx=max(max(X));
errors_lin = []
errors_k = []

noisefactor =1;

noise = noisefactor*maxx; % sd for Gaussian noise


Xn = X; 
for i=1:N;
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt = Xtest1; 
for i=1:size(Xtest1,1);
  randn('state', N+i);
  Xnt(i,:) = Xtest1(i,:) + noise*randn(1,dim);
end

%
Xtr = X(60:1:end,:);
XVal = X(1:1:59,:);
[N, dim]=size(XVal);

sig2 =dim*mean(var(XVal)); % rule of thumb
sigmafactor = 1;
sig2=sig2*sigmafactor;
[lam_lin,U_lin] = pca(XVal);

% kernel PCA
[lam,U] = kpca(XVal,'RBF_kernel',sig2,[],'eig',240); 
[lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);

npcs = [2.^(0:7) 59];
lpcs = length(npcs);

digs=[0]; ndig=length(digs);
m=2; % Choose the mth data for each digit 

Xdt=zeros(ndig,dim);

for k=1:lpcs;
 nb_pcs=npcs(k); 
 Ud=U(:,(1:nb_pcs)); %lamd=lam(1:nb_pcs);
 Ud_lin=U_lin(:,(1:nb_pcs)); lamd=lam_lin(1:nb_pcs);
 for i=1:ndig
     xt=Xnt(i,:);
     proj_lin=xt*Ud_lin;
     Xdt(i,:) = preimage_rbf(XVal,sig2,Ud,xt,'denoise');
     Xdt_lin(i,:) = proj_lin*Ud_lin';
     errors_lin = [errors_lin, round(sum(sum((Xdt_lin - Xtest2(1,:)).^2)),1)];
     errors_k = [errors_lin, round(sum(sum((Xdt - Xtest2(1,:)).^2)),1)];
 end
 
end