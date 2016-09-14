%% 
% This is a testing file for three different conditions
addpath(genpath(pwd));
load('badexample.mat');
ytest = -38;
xnew = X_withnew(end,:);
%% GLMNET

%%
% The following code gives the TRUE lasso model by glmnet
fit = glmnet(X_withnew,[Y;ytest],[],options);E = find(fit.beta);
fprintf(1, 'The model at %.2f is\n', ytest);
disp(E);

%%
% from testing, we can see that the support stays the same in the range [-39.7, -36.5]

%% Polyhedron conditions Ay<b
%%
% The following code gives the range of such support from Ay<b condition

beta = glmnetCoef(glmnet(X_withnew,[Y;ytest],[],options));
beta=beta(2:p+1);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
X_minusE = X_withnew(:,setxor(E,1:p));
X_E = X_withnew(:,E);
% accelerate computation
xesquareinv = (X_E'*X_E)\eye(length(E));
P_E = X_E*xesquareinv*X_E';
temp = X_minusE'*pinv(X_E')*Z_E;
a0=X_minusE'*(eye(m+1)-P_E)./lambdain;
% calculate the inequalities for fitting.
A = [a0;
    -a0;
    -diag(Z_E)*xesquareinv*X_E'];
b = [ones(p-length(E),1)-temp;
    ones(p-length(E),1)+temp;
    -lambdain*diag(Z_E)*xesquareinv*Z_E];
[smin,smax] = solveInt(A,b,Y);
fprintf(1,'The range of this model is [%.2f, %.2f]\n',smin,smax);

%%
% this range is slightly shorter than the TRUE range, but only up to
% computational error

%% |X'*resid|<lambda

%%
% First we try solving for the range that the support stays the same
beta = glmnetCoef(glmnet(X_withnew,[Y;ytest],[],options));
beta=beta(2:p+1);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
X_minusE = X_withnew(:,setxor(E,1:p));
X_E = X_withnew(:,E);

pinvxe=pinv(X_E);
betalast = pinvxe(:,end); 
betaincrement = zeros(p,1); betaincrement(E) = betalast;
yfitincrement = X_withnew*betaincrement;
yfit = X_withnew*beta;
A = X_withnew'*([Y;ytest]-yfit);
left=xnew'-X_withnew'*yfitincrement;
rightplus=lambdain-A;
rightminus=-lambdain-A;

deltaplus = rightplus./left;
deltaplus(deltaplus<=0)=inf;
[minstepplus,minstepplusind] = min(deltaplus);
deltaminus = rightminus./left;
deltaminus(deltaminus<=0)=inf;
[minstepminus,minstepminusind] = min(deltaminus);
if minstepplus<=minstepminus
    step = minstepplus;
    stepind = minstepplusind;
    stepsign = 1;
else
    step = minstepminus;
    stepind = minstepminusind;
    stepsign = -1;
end

%%
% The first to add in/drop outof the model is number 1112, adding into the
% model, with positive sign, at y=
disp(ytest + step);
%%
% but this is far beyond the TRUE range of support. Take the model from 
% ytest = -38, we check the inequalities at y = -10, which should be out 
% of the model. The following code gives the features that violates 
% $$ |X'*Residue|<\lambda  $$
% given the support and signs we obtained at ytest=38
beta = zeros(p,1);
beta(E) = pinvxe*[Y;-10] - xesquareinv*Z_E*lambdain;
t=X_withnew'*([Y;-10]-X_withnew*beta);

%%
% Inequalities with negative signs are
find(t<-lambdain+1E-6)

%%
% Inequalities with positive signs are
find(t>lambdain-1E-6)

%%
% This shows here at y=-10 shares the same model as at ytest=-38, which
% is not true, because we know the true model should be:

fit = glmnet(X_withnew,[Y;-10],[],options);E = find(fit.beta);
fprintf(1, 'The model at %.2f is\n', -10);
disp(E)