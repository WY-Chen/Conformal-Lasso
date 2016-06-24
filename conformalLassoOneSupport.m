%% Conformal inference
% Method: run conformal inference on a data set with lasso.
% only run Lasso by Lars once, and calculate boundary of y,  
% then run with known model for the points in ytrial. 
%% Method
function [yconf,supportcoverage,modelsize] = conformalLassoOneSupport(X,Y,xnew,alpha,ytrial)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    trail set for y

% prepare for fitting
addpath(genpath(pwd));
n = length(ytrial);

X_withnew = [X;xnew];
Pi_trial = ones(1,n);
[m,p] = size(X);

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = 1.0;                % Lasso (no L2 norm penalty)
options.thresh = 1E-12;

% fit the first model and calculate support.
fit = cvglmnet(X,Y,[],options);
lambda = fit.lambda_1se*m;
beta = cvglmnetCoef(fit);
beta = beta(2:p+1);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
X_minusE = X_withnew(:,setxor(E,1:p));
X_E = X_withnew(:,E);
P_E = X_E*((X_E'*X_E)\X_E');
% calculate the inequalities for fitting. 
A = [X_minusE'*(eye(m+1)-P_E)./lambda;
    -X_minusE'*(eye(m+1)-P_E)./lambda;
    -diag(Z_E)*((X_E'*X_E)\X_E')];
b = [ones(p-length(E),1)-X_minusE'*pinv(X_E')*Z_E;
    ones(p-length(E),1)+X_minusE'*pinv(X_E')*Z_E;
    -lambda*diag(Z_E)*((X_E'*X_E)\Z_E)];
jumpcount=0;
fprintf('\tPrediction point is %2.2f\n', xnew*beta)
if isempty(E)
    yconf = [];
    supportcoverage = 0;
    return
end

for i = 1:n
    y = ytrial(i);
    if not(all(A*[Y;y]<=b))
        jumpcount=jumpcount+1;
%         disp(max(A*[Y;y]-b))
        continue
    end
    beta = zeros(p,1);
    beta(E) = pinv(X_E)*[Y;y] - lambda*((X_E'*X_E)\Z_E);  
    yfit = X_withnew*beta;
    Resid = abs(yfit - [Y;y]);
    Pi_trial(i) = sum(Resid<=Resid(end))/(m+1);
end
yconf = ytrial(Pi_trial<=1-alpha);
supportcoverage = (n-jumpcount)/n;
modelsize = length(E);
end
