%% Costumised Lasso with GLMnet
% return polyhedron
% If called with lambdain=='CV', do cross validation version.
% if called with given lambdain, use this lambda to fit lasso.
%% Method
function [beta,A,b,lambda] = lassoSupport(X,Y,X_withnew,lambdain)

% prepare for fitting
addpath(genpath(pwd));
[m,p] = size(X);
[mnew,~] = size(X_withnew);

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = 1.0;                % Lasso (no L2 norm penalty)
options.thresh = 1E-12;

% fit the lasso and calculate support.
if ~isequal(lambdain,'CV')
    lambda = lambdain;
    beta = lasso(X,Y,'Lambda',lambda/m,'Standardize',0,'RelTol',1E-12);
else
    fit = cvglmnet(X,Y,[],options);
    lambda = fit.lambda_1se*m;
    beta = cvglmnetCoef(fit);
    beta = beta(2:p+1);
end

E = find(beta);
Z = sign(beta);
Z_E = Z(E);
X_minusE = X_withnew(:,setxor(E,1:p));
X_E = X_withnew(:,E);
P_E = X_E*((X_E'*X_E)\X_E');
% calculate the inequalities for fitting. 
A = [X_minusE'*(eye(mnew)-P_E)./lambda;
    -X_minusE'*(eye(mnew)-P_E)./lambda;
    -diag(Z_E)*((X_E'*X_E)\X_E')];
b = [ones(p-length(E),1)-X_minusE'*pinv(X_E')*Z_E;
    ones(p-length(E),1)+X_minusE'*pinv(X_E')*Z_E;
    -lambda*diag(Z_E)*((X_E'*X_E)\Z_E)];