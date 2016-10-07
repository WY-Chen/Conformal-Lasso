%% Conformal inference
% Method: run conformal inference on a data set with LTS lasso.
% fit a LTS lasso to (X,Y) and claim that the support does not change for
% models with Xnew included, and Xnew does not affect models where it is 
% excluded. 
%% Method
function [yconf,modelsize] = conformalLTSLassoOneSupp(X,Y,xnew,alpha,ytrial)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    trail set for y

% prepare for fitting
addpath(genpath(pwd));
X_withnew = [X;xnew];
[m,p] = size(X);
n = length(ytrial);
Linoptions = optimset('Display','off');

% fit the model
yconfidx = [];
[beta,A,b,lambda,H] = LTSlassoSupport(X,Y,xnew);
H=sort(H);
h= length(H);
supportmax = linprog(-1,A(:,h+1),realmin+b-A(:,1:h)*Y(H),[],[],[],[],[],Linoptions);
supportmin = linprog(1,A(:,h+1),realmin+b-A(:,1:h)*Y(H),[],[],[],[],[],Linoptions);
supportmin = min(supportmin,supportmax);
supportmax = max(supportmin,supportmax);
Supp=find(ytrial>=supportmin & ytrial<=supportmax);
disp([supportmin supportmax])

h = waitbar(0,'Please wait...');
for i=1:n
    y = ytrial(i); 
    Resid = abs(X_withnew*beta - [Y;y]);
    Pi_trial = sum(Resid<=Resid(end))/(m+1);
    if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
        yconfidx = [yconfidx i];
    end
    waitbar(i/n,h)
end
close(h)


yconf  = ytrial(yconfidx);
modelsize = length(find(beta));
end
