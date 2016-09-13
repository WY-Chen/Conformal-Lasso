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
stepsize = ytrial(2)-ytrial(1);

% Initial and Stopping range
yinit = min(ytrial);
yterm = max(ytrial);

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

% Fit initial Lasso
beta = glmnetCoef(glmnet(X_withnew,[Y;yinit],[],options));
beta=beta(2:p+1);
E = find(beta);
Z = sign(beta);
X_E = X_withnew(:,E);
% accelerate computation
pinvxe=pinv(X_E);
betalast = pinvxe(:,end);
betaincrement = zeros(p,1);
betaincrement(E) = betalast;
yfitincrement = X_withnew*betaincrement;
yfit = X_withnew*beta;
A = X_withnew'*([Y;0]-yfit);
left=xnew'-X_withnew'*yfitincrement;
rightplus=lambdain-A-xnew'*yinit;
rightminus=-lambdain-A-xnew'*yinit;

supportcounter = 1;
ysmax=yinit;
yconf = []; modelsize=0;
while 1
    % solve for the next model
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
    ysmin = ysmax;
    ysmax = ysmax + step;
   
    % solve for conformal
    thistrial = ytrial(ytrial>ysmin & ytrial<ysmax);
    modelsize = modelsize + length(E)*length(thistrial);
    for y=thistrial
        yfit = yfit + yfitincrement*stepsize;
        Resid = abs([Y;y]-yfit);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
            yconf = [yconf y];
        end
    end
    % end of solving
    
    if ismember(stepind,E)
        E = E(E~=stepind);
        Z(stepind)=0;
        Z_E = Z(E);
    else
        E = sort([E;stepind]);
        Z(stepind)=stepsign;
        Z_E=Z(E);
    end
    if ysmax > yterm
        break
    end
    if isempty(E)
        fprintf(2,'E empty \n');
    end
    X_E = X_withnew(:,E);
    xesquareinv = (X_E'*X_E)\eye(length(E));
    pinvxe=pinv(X_E);
    betalast = pinvxe(:,end);
    betaincrement = zeros(p,1);
    betaincrement(E) = betalast;
    yfitincrement = X_withnew*betaincrement;
    beta = zeros(p,1);
    beta(E) = pinvxe*[Y;ysmax] - lambdain*xesquareinv*Z_E;
    yfit = X_withnew*beta;
    % Robooting
    ineqviolated = find(abs(X_withnew'*([Y;ysmax]-yfit))>=lambdain);
    if ~isequal(ineqviolated, E)
        beta = glmnetCoef(glmnet(X_withnew,[Y;ysmax],[],options));
        beta=beta(2:p+1);
        E = find(beta);
        Z = sign(beta);
        X_E = X_withnew(:,E);
        % accelerate computation
        pinvxe=pinv(X_E);
        betalast = pinvxe(:,end);
        betaincrement = zeros(p,1);
        betaincrement(E) = betalast;
        yfitincrement = X_withnew*betaincrement;
        yfit = X_withnew*beta;
    end
    
    A = X_withnew'*([Y;0]-yfit);
    left=xnew'-X_withnew'*yfitincrement;
    rightplus=lambdain-A-xnew'*ysmax;
    rightminus=-lambdain-A-xnew'*ysmax;
    
    supportcounter = supportcounter+1;
end
modelsize = modelsize/n;
end
Contact GitHub API Training Shop Blog About
© 2016 GitHub, Inc. Terms Privacy Security Status Help