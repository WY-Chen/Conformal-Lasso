%% Conformal inference
% Method: run conformal inference on a data set with lasso.
% only run Elastic net by Lars once, and calculate boundary of y,  
% then run with known model for the points in ytrial. 
%% Method
function [yconf,modelsize] = conformalENOneSupp(X,Y,xnew,alpha,ytrial)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    trail set for y

% prepare for fitting
addpath(genpath(pwd));
X_withnew = [X;xnew];
[m,p] = size(X);
n = length(ytrial);
normmix = 0.8;

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = normmix;                 % Run Elastic net
options.thresh = 1E-12;
Linoptions = optimset('Display','off');

% fit the first model and calculate support.
yconfidx = [];
[beta,A,b,lambda] = ENSupport(X,Y,X_withnew,normmix);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
fprintf('\tPrediction point is %2.2f\n', xnew*beta)
supportmax = linprog(-1,A(:,m+1),realmin+b-A(:,1:m)*Y,[],[],[],[],[],Linoptions);
supportmin = linprog(1,A(:,m+1),realmin+b-A(:,1:m)*Y,[],[],[],[],[],Linoptions);
supportmin = min(supportmin,supportmax);
supportmax = max(supportmin,supportmax);
disp([supportmin supportmax]);

h = waitbar(0,'Please wait...');
Supp=find(ytrial>=supportmin & ytrial<=supportmax);
gamma = 2*(1-normmix)*lambda/normmix;

for i=Supp
    y = ytrial(i); 
    beta = zeros(p,1);
    X_E = X_withnew(:,E);
    c = (1+gamma).^(-1/2);
    beta(E) = (X_E'*X_E+eye(length(E))*gamma)\X_E'*[Y;y]...
        - 2*c*lambda*((X_E'*X_E+eye(length(E))*gamma)\Z_E);  
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
if isempty(yconf)
    yconf = ytrial;  % rather return the whole interval
    fprintf('WARNING: returned the whole interval.\n');
    modelsize = length(E);
    return
end
modelsize = length(E);
end
