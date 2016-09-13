%% Conformal inference
% Method: run conformal inference on a data set with lasso.
% only run Lasso by Lars once, and calculate boundary of y,  
% then run with known model for the points in ytrial. 
%% Method
function [yconf,modelsize] = conformalLassoOneSupp(X,Y,xnew,alpha,ytrial,lambdain)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    trail set for y

if nargin==5
    lambda = 'CV';
else
    lambda = lambdain;
end

% prepare for fitting
addpath(genpath(pwd));
X_withnew = [X;xnew];
[m,p] = size(X);
n = length(ytrial);

% Tune GLMNET
Linoptions = optimset('Display','off');

% fit the first model and calculate support.
yconfidx = [];
[beta,A,b,lambda] = lassoSupport(X,Y,X_withnew,lambda);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
fprintf('\tPrediction point is %2.2f\n', xnew*beta)
supportmax = linprog(-1,A(:,m+1),b-A(:,1:m)*Y,[],[],[],[],[],Linoptions);
supportmin = linprog(1,A(:,m+1),b-A(:,1:m)*Y,[],[],[],[],[],Linoptions);

h = waitbar(0,'Please wait...');
Supp=find(ytrial>=supportmin & ytrial<=supportmax);
if isempty(Supp)
    yconf = ytrial;  % rather return the whole interval
    fprintf('WARNING: returned the whole interval.\n');
    modelsize = length(E);
    close(h)
    return
end
for i=Supp
    y = ytrial(i); 
    beta = zeros(p,1);
    X_E = X_withnew(:,E);
    beta(E) = pinv(X_E)*[Y;y] - lambda*((X_E'*X_E)\Z_E);  
    yfit = X_withnew*beta;
    Resid = abs(yfit - [Y;y]);
    Pi_trial = sum(Resid<=Resid(end))/(m+1);
    if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
        yconfidx = [yconfidx i];
    end
    waitbar(i/n,h)
end
close(h)
yconf  = ytrial(yconfidx);
modelsize = length(E);
end
