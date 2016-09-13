%% Costumised Lasso
% return polyhedron
%% Method
function [beta,A,b,lambda,H] = LTSlassoSupport(X,Y,xnew,tau,lambdain)

% prepare for fitting
addpath(genpath(pwd));
[~,p] = size(X);

% fit the lasso and calculate support.
[beta,H,lambda] = LTSlasso(X,Y,lambdain,tau);        % decide whether to use glmnet here
h=length(H);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
X_withnew = [X;xnew];
X_minusE = X_withnew(H,setxor(E,1:p));
X_E = X_withnew(H,E);
% accelerate the computation
xesquareinv = (X_E'*X_E)\eye(length(E));
P_E = X_E*xesquareinv*X_E';
temp = X_minusE'*pinv(X_E')*Z_E;
a0=X_minusE'*(eye(h)-P_E)./lambda;
% calculate the inequalities for fitting. 
A = [a0;
    -a0;
    -diag(Z_E)*xesquareinv*X_E'];
b = [ones(p-length(E),1)-temp;
    ones(p-length(E),1)+temp;
    -lambda*diag(Z_E)*xesquareinv*Z_E];