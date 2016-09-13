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

%% Initialization with the first point
yconfidx = []; 
supportcounter = 1;
beta = glmnetCoef(glmnet(X_withnew,[Y;ytrial(1)],[],options));
beta=beta(2:p+1);
E = find(beta);
Z = sign(beta);
X_E = X_withnew(:,E);
pinvxe=pinv(X_E);
betalast = pinvxe(:,end);
betaincrement = zeros(p,1);
betaincrement(E) = betalast;
yfitincrement = X_withnew*betaincrement;
stepsize = ytrial(2)-ytrial(1);
yfit = X_withnew*beta;

for i=2:n
    y = ytrial(i);
    yfit = yfit + yfitincrement*stepsize;
    ineqviolatedplus = find(X_withnew'*([Y;y]-yfit)>lambdain);
    ineqviolatedminus = find(X_withnew'*([Y;y]-yfit)<-lambdain);
    if ~isequal([ineqviolatedminus;ineqviolatedplus],E)
        beta = glmnetCoef(glmnet(X_withnew,[Y;y],[],options));
        beta = beta(2:p+1);
        E = find(beta);
        X_E = X_withnew(:,E);
        pinvxe=pinv(X_E);
        betalast = pinvxe(:,end);
        betaincrement = zeros(p,1);
        betaincrement(E) = betalast;
        yfitincrement = X_withnew*betaincrement;
        yfit = X_withnew*beta;
        supportcounter = supportcounter +1;
    end
    Resid = abs(yfit - [Y;y]);
    Pi_trial = sum(Resid<=Resid(end))/(m+1);
    if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
        yconfidx = [yconfidx i];
    end
 
    modelsizes(i) = length(E);
end
% close(h)
modelsize = mean(modelsizes);
yconf  = ytrial(yconfidx);

% Plots
plotFlag=0;  % change to 0 to turn off
if plotFlag == 1
    subplot(1,2,1)
    boxplot(yconf);
    title('Spread of Yconf')
    subplot(1,2,2)
    plot(ytrial,modelsizes);
    title('Model size vs. Ytrial')
    hold on, plot(ytrial(yconfidx), modelsizes(yconfidx), 'r.')
    hold off
end
end