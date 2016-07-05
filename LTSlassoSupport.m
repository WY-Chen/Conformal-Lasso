%% Costumised Lasso with GLMnet
% return polyhedron
%% Method
function [beta,A,b,lambda,H] = LTSlassoSupport(X,Y,xnew)

% prepare for fitting
addpath(genpath(pwd));
[m,p] = size(X);

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = 1.0;                % Lasso (no L2 norm penalty)
options.thresh = 1E-12;

% fit the lasso and calculate support.
fit = cvglmnet(X,Y,[],options);
lambda = fit.lambda_1se*0.9;        % Need to consider other ways to choose

% fit the lasso and calculate support.
[beta,H] = LTSlasso(X,Y,lambda,0.9);
h=length(H);
lambda = lambda*m;
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
X_withnew = [X;xnew];
X_minusE = X_withnew([H;m+1],setxor(E,1:p));
X_E = X_withnew([H;m+1],E);
P_E = X_E*((X_E'*X_E)\X_E');
% calculate the inequalities for fitting. 
A = [X_minusE'*(eye(h+1)-P_E)./lambda;
    -X_minusE'*(eye(h+1)-P_E)./lambda;
    -diag(Z_E)*((X_E'*X_E)\X_E')];
b = [ones(p-length(E),1)-X_minusE'*pinv(X_E')*Z_E;
    ones(p-length(E),1)+X_minusE'*pinv(X_E')*Z_E;
    -lambda*diag(Z_E)*((X_E'*X_E)\Z_E)];