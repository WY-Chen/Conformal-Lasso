%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new point is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,supportcoverage,modelsize] = conformalLassoAllSupp(X,Y,xnew,alpha,ytrial)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% stepsize  stepsize of searching for upper and lower bound of interval

% prepare for fitting
addpath(genpath(pwd));
X_withnew = [X;xnew];
[m,p] = size(X);
n = length(ytrial);
modelsizes = zeros(1,n);

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = 1.0;                % Lasso (no L2 norm penalty)
options.thresh = 1E-12;

% Build confidence interval
yconf = [];
[beta,A,b,lambda] = lassoSupport(X_withnew,[Y;ytrial(1)],X_withnew);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
supportcounter = 1;
fprintf('\tPrediction point is %2.2f\n', xnew*beta)

h = waitbar(0,'Please wait...');
for i = 1:n
    y = ytrial(i);
    if all(A*[Y;y]-b<=0)
        beta = zeros(p,1);
        X_E = X_withnew(:,E);
        beta(E) = pinv(X_E)*[Y;y] - lambda*((X_E'*X_E)\Z_E);  
        yfit = X_withnew*beta;
        Resid = abs(yfit - [Y;y]);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=1-alpha
            yconf = [yconf y];
        end
    else 
        [beta,A,b,lambda] = lassoSupport(X_withnew,[Y;y],X_withnew);
        E = find(beta);
        Z = sign(beta);
        Z_E = Z(E);
        yfit = X_withnew*beta;
        Resid = abs(yfit - [Y;y]);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=1-alpha
            yconf = [yconf y];
        end
        supportcounter = supportcounter+1;
    end   
    modelsizes(i) = length(E);
    % waitbar
    waitbar(i/n,h,sprintf('Number of Lasso support computed %d',supportcounter))
end
close(h)
supportcoverage = 1;
modelsize = mean(modelsizes);
% plot(modelsizes)
end