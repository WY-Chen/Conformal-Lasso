%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new point is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize,supportcounter] = conformalLassoCtnFit(X,Y,xnew,alpha,ytrial,lambdain)
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

% Initialization with first point
yinit = ytrial(1);
% Fit initial Lasso
beta = glmnetCoef(glmnet(X_withnew,[Y;yinit],[],options));
beta=beta(2:p+1);
E = find(beta);
Z = sign(beta);
Z_E=Z(E);
X_E = X_withnew(:,E);
% accelerate computation
pinvxe=pinv(X_E);
xesquareinv = (X_E'*X_E)\eye(length(E));
yfit = X_withnew*beta;

%% fit
yconf = [];
modelsizes =zeros(1,m);modelsizes(1)=length(E);
sc=1;
for i=2:n
    y=ytrial(i);
    beta(E) = pinvxe*[Y;y] - xesquareinv*Z_E*lambdain;
    yfit = X_withnew*beta;
    ineqleft = X_withnew'*([Y;y]-yfit);
    ineqplus = find(ineqleft>lambdain);
    ineqminus = find(ineqleft<-lambdain);
    if ~isequal(E,sort([ineqplus;ineqminus]))
        E = sort([ineqplus;ineqminus]);
        Z = zeros(p,1);
        Z(ineqplus)=1;
        Z(ineqminus)=-1;
        Z_E=Z(E);
        X_E = X_withnew(:,E);
        % accelerate computation
        pinvxe=pinv(X_E);
        xesquareinv = (X_E'*X_E)\eye(length(E));
        beta = zeros(p,1);
        beta(E) = pinvxe*[Y;y] - xesquareinv*Z_E*lambdain;
        yfit = X_withnew*beta;
        sc = sc+1;
        fprintf(2,'Refiting: %d\n',length(E));
        
%         % Robooting
%         ineqviolated = find(abs(X_withnew'*([Y;y]-yfit))>=lambdain);
%         if ~isequal(ineqviolated, E)
%             beta = glmnetCoef(glmnet(X_withnew,[Y;y],[],options));
%             beta=beta(2:p+1);
%             E = find(beta);
%             Z = sign(beta);
%             Z_E = Z(E);
%             X_E = X_withnew(:,E);
%             % accelerate computation
%             pinvxe=pinv(X_E);
%             xesquareinv = (X_E'*X_E)\eye(length(E));
%             yfit = X_withnew*beta;
%             fprintf(2,'Rebooting: %d\n',length(E));
%         end
    end
    
    Resid = abs([Y;y]-yfit);
    Pi_trial = sum(Resid<=Resid(end))/(m+1);
    if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
        yconf = [yconf y];
    end
    
    
    
    modelsizes(i)=length(E);
end

modelsize = mean(modelsizes);
supportcounter = sc;
end