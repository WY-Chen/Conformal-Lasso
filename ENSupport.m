%% Costumised Elastic Net with GLMnet
% return polyhedron
%% Method
function [beta,A,b,lambda] = ENSupport(X,Y,X_withnew,normmix)

% prepare for fitting
addpath(genpath(pwd));
[m,p] = size(X);
[mnew,pnew] = size(X_withnew);
if ~pnew==p
    fprintf('ERROR: new X matrix dimension')
    return
end

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = normmix;              % Elastic net
options.thresh = 1E-12;

% fit the lasso and calculate support.
fit = cvglmnet(X,Y,[],options);
lambda = fit.lambda_1se*m;
gamma = 2*(1-normmix)*fit.lambda_1se*m/normmix;
beta = cvglmnetCoef(fit);
beta = beta(2:p+1);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
X_minusE = X_withnew(:,setxor(E,1:p));
X_E = X_withnew(:,E);
P_E = X_E*((X_E'*X_E+eye(length(E))*gamma)\X_E');
X_ETP = (X_E*X_E'+eye(mnew)*gamma)\X_E;
% calculate the inequalities for fitting. 
A = [X_minusE'*(eye(mnew)-P_E)./lambda;
    -X_minusE'*(eye(mnew)-P_E)./lambda;
    -diag(Z_E)*((X_E'*X_E+eye(length(E))*gamma)\X_E')];
b = [ones(p-length(E),1)-X_minusE'*X_ETP*Z_E;
    ones(p-length(E),1)+X_minusE'*X_ETP*Z_E;
    -lambda*diag(Z_E)*((X_E'*X_E+eye(length(E))*gamma)\Z_E)];