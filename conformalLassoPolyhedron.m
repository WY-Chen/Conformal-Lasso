%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new point is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize,supportcounter] = conformalLassoPolyhedron(X,Y,xnew,alpha,ytrial,lambdain)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% stepsize  stepsize of searching for upper and lower bound of interval
% call it with lambda to use the fixed lambda method
% call it without lambda to use the cv lambda method

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
options.nlambda = 1;
options.lambda = lambdain/m;

%% Initialization with the first point
yconfidx = []; 
supportcounter = 0;

% Fit initial Lasso
beta = glmnetCoef(glmnet(X_withnew,[Y;ytrial(1)],[],options));
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
pinvxe=pinv(X_E);
betalast = pinvxe(:,end);
betaincrement = zeros(p,1);
betaincrement(E) = betalast;
yfitincrement = X_withnew*betaincrement;
yfit = X_withnew*beta;
[supportmin,supportmax] = solveInt(A,b,Y);
% avoid null model
if isempty(E)
    supportmin = inf;
end
modelsize=length(E);
lengthE=length(E);
supportcounter = 1;
if isempty(E)
    supportmin = inf;
end

for i = 2:n
    y = ytrial(i);
    if supportmin< y && supportmax >y
        stepsize = ytrial(i)-ytrial(i-1);
        yfit = yfit + yfitincrement*stepsize;
    else
        Ineq_violated = [find(X_withnew'*(X_withnew*beta-[Y;y])>lambdain);...
            find(X_withnew'*(X_withnew*beta-[Y;y])<-lambdain)];
        E=sort(Ineq_violated);
        Z=zeros(p,1);
        Z(X_withnew'*(X_withnew*beta-[Y;y])>lambdain)=1;
        Z(X_withnew'*(X_withnew*beta-[Y;y])<-lambdain)=-1;
        Z_E=Z(E);

        lengthE=length(E);
        X_E = X_withnew(:,E);
        X_minusE = X_withnew(:,setxor(E,1:p));
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
        supportcounter = supportcounter+1;
        [supportmin,supportmax] = solveInt(A,b,Y);
        if isempty(E)
            supportmin = inf;
        end
        pinvxe=pinv(X_E);
        betalast = pinvxe(:,end);
        betaincrement = zeros(p,1);
        betaincrement(E) = betalast;
        yfitincrement = X_withnew*betaincrement;
        yfit = X_withnew*beta;
    end
    Resid = abs(yfit - [Y;y]);
    Pi_trial = sum(Resid<=Resid(end))/(m+1);
    if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
        yconfidx = [yconfidx i];
    end
    modelsizes(i) = length(E);
end
modelsize = mean(modelsizes);
yconf  = ytrial(yconfidx);
end