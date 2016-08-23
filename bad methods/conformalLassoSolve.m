%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new point is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize,supportcounter] = conformalLassoSolve(X,Y,xnew,alpha,ytrial,lambdain)
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
supportcounter = 0;

% The first point
y=ytrial(1);
yconfmin=inf;
yconfmax=-inf;

while 1
    beta = glmnetCoef(glmnet(X_withnew,[Y;y],[],options));
    beta=beta(2:p+1);
    E = find(beta);
    Z = sign(beta);
    Z_E = Z(E);
    X_minusE = X_withnew(:,setxor(E,1:p));
    X_E = X_withnew(:,E);
    % accelerate computation
    % accelerate the computation
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
    modelsizes = [modelsizes,length(E)];
    [~,supportmax] = solveInt(A,b,Y);
    pinvxe=pinv(X_E);
    betalast = pinvxe(:,end);
    betaincrement = zeros(p,1);
    betaincrement(E) = betalast;
    D = X_withnew*betaincrement;
    betafix = zeros(p,1);
    betafix(E) = pinvxe(:,1:end-1)*Y - lambdain*xesquareinv*Z_E;
    C = X_withnew*betafix;
    
    res = sort((C(end)-C+[Y;y])./(1+D-D(end)));
    y2max = res(ceil((1-alpha)*(m+1)));


    yconfmax=max(yconfmax,y2max);
    
    % new y value
    if supportmax<y2max
        fprintf('wont let you\n');
    end
    y = min(supportmax,y2max)+1E-3;
    
    if y>ytrial(end)
        break;
    end
end
yconf = [yconfmin,yconfmax];
modelsize = mean(modelsizes);



end