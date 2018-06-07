function [errlist] = gridSearch(sigma_params,gam_params, X, Y, Xt, Yt, plot)
% Xt and Yt stand for test
% returns missclasification errors using an RBF kernel, and the
% given parameters and train/test sets
% ----------- output --------------
% errlist: N(sigmas) * M(gammas) matrix with missclassif results

type = 'c';
errlist = [];


for sig2=sigma_params
    for gam=gam_params
        disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
        [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

        % Obtain the output of the trained classifier
        [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
        err = sum(Yht~=Yt)./size(Yt,1); errlist=[errlist; err];
        if (plot == true)
            fprintf('\n With gamma = %d \n sigma = %.f%', gam, sig2)
            fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err * size(Yt,1), err*100)
        end
        
        % Plot the decision boundary of a 2-d LS-SVM classifier
        if (plot == true)
            plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
            disp('Press any key to continue...'), pause,  
        end

    end
end
cols = length(sigma_params);
rows = length(gam_params);
errlist = reshape(errlist, [rows, cols])';  % transposed so rows are sigma
end

