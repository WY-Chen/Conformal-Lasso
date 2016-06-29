%% Conformal inference
% Method: run conformal inference on a data set with lasso.
% only run Lasso by Lars once, and calculate boundary of y,  
% then run with known model for the points in ytrial. 
%% Method
function [yconf,supportcoverage,modelsize] = conformalLassopredMultSupp(X,Y,xnew,alpha,stepsize)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% stepsize  stepsize of searching for upper and lower bound of interval

% prepare for fitting
addpath(genpath(pwd));
X_withnew = [X;xnew];
[m,p] = size(X);
modelsize=0;
yconf = [];

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = 1.0;                % Lasso (no L2 norm penalty)
options.thresh = 1E-12;

% fit the first model and calculate support.
fit = cvglmnet(X,Y,[],options);
lambda = fit.lambda_1se*m;
beta = cvglmnetCoef(fit);
beta = beta(2:p+1);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
X_minusE = X_withnew(:,setxor(E,1:p));
X_E = X_withnew(:,E);
P_E = X_E*((X_E'*X_E)\X_E');
% calculate the inequalities for fitting. 
A = [X_minusE'*(eye(m+1)-P_E)./lambda;
    -X_minusE'*(eye(m+1)-P_E)./lambda;
    -diag(Z_E)*((X_E'*X_E)\X_E')];
b = [ones(p-length(E),1)-X_minusE'*pinv(X_E')*Z_E;
    ones(p-length(E),1)+X_minusE'*pinv(X_E')*Z_E;
    -lambda*diag(Z_E)*((X_E'*X_E)\Z_E)];
fprintf('\tPrediction point is %2.2f\n', xnew*beta)
if isempty(E)
    fprintf('ERROR: LASSO gives NULL model\n')
    yconf = [];
    supportcoverage = 0;
    return
end

% setting the initial guess
yinit = cvglmnetPredict(fit,xnew);

if not(all(A*[Y;yinit]-b<=0))
    fprintf('ERROR: Initial Guess not in support\n')
    yconf = [];
    supportcoverage = 0;
    return
end

% Build confidence interval around the initial guess 
yconflower = yinit; yconfupper = yinit; temp = yinit;
while all(A*[Y;temp]-b<=0)
    beta = zeros(p,1);
    beta(E) = pinv(X_E)*[Y;temp] - lambda*((X_E'*X_E)\Z_E);  
    yfit = X_withnew*beta;
    Resid = abs(yfit - [Y;temp]);
    Pi_trial = sum(Resid<=Resid(end))/(m+1);
    if Pi_trial<=1-alpha
        yconflower = temp;
        yconf = [yconf temp];
    end
    temp = temp - stepsize;
end
temp = yinit;
while all(A*[Y;temp]-b<=0)
    beta = zeros(p,1);
    beta(E) = pinv(X_E)*[Y;temp] - lambda*((X_E'*X_E)\Z_E);  
    yfit = X_withnew*beta;
    Resid = abs(yfit - [Y;temp]);
    Pi_trial = sum(Resid<=Resid(end))/(m+1);
    if Pi_trial<=1-alpha
        yconfupper = temp;
        yconf = [yconf temp];
    end
    temp = temp + stepsize;
end
modelsize = length(E);

% the secondary runs
yconflowerold = yconflower;
yconfupperold = yconfupper;
iterl = 1;
while 1
    temp = yconflower - stepsize;
    % get the new support
    fit = cvglmnet(X_withnew,[Y;temp],[],options);
    lambda = fit.lambda_1se*m;
    beta = cvglmnetCoef(fit);
    beta = beta(2:p);
    E = find(beta);
    Z = sign(beta);
    Z_E = Z(E);
    X_minusE = X_withnew(:,setxor(E,1:p));
    X_E = X_withnew(:,E);
    P_E = X_E*((X_E'*X_E)\X_E');
    % calculate the inequalities for fitting. 
    A = [X_minusE'*(eye(m+1)-P_E)./lambda;
        -X_minusE'*(eye(m+1)-P_E)./lambda;
        -diag(Z_E)*((X_E'*X_E)\X_E')];
    b = [ones(p-length(E),1)-X_minusE'*pinv(X_E')*Z_E;
        ones(p-length(E),1)+X_minusE'*pinv(X_E')*Z_E;
        -lambda*diag(Z_E)*((X_E'*X_E)\Z_E)];
    while all(A*[Y;temp]-b<=0)
        beta = zeros(p,1);
        beta(E) = pinv(X_E)*[Y;temp] - lambda*((X_E'*X_E)\Z_E);  
        yfit = X_withnew*beta;
        Resid = abs(yfit - [Y;temp]);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=1-alpha
            yconflower = temp;
            yconf = [yconf temp];
        end
        temp = temp - stepsize;
    end
    if yconflower == yconflowerold
        break;
    end    
    iterl = iterl + 1;
    fprintf('\t\tSupport Iteration Left: %d\n',iterl)
    yconflowerold = yconflower;
    modelsize = modelsize+length(E);
end
iterr = 1;
while 1
    temp = yconfupper + stepsize;
    % get the new support
    fit = cvglmnet(X_withnew,[Y;temp],[],options);
    lambda = fit.lambda_1se*m;
    beta = cvglmnetCoef(fit);
    beta = beta(2:p);
    E = find(beta);
    Z = sign(beta);
    Z_E = Z(E);
    X_minusE = X_withnew(:,setxor(E,1:p));
    X_E = X_withnew(:,E);
    P_E = X_E*((X_E'*X_E)\X_E');
    % calculate the inequalities for fitting. 
    A = [X_minusE'*(eye(m+1)-P_E)./lambda;
        -X_minusE'*(eye(m+1)-P_E)./lambda;
        -diag(Z_E)*((X_E'*X_E)\X_E')];
    b = [ones(p-length(E),1)-X_minusE'*pinv(X_E')*Z_E;
        ones(p-length(E),1)+X_minusE'*pinv(X_E')*Z_E;
        -lambda*diag(Z_E)*((X_E'*X_E)\Z_E)];
    while all(A*[Y;temp]-b<=0)
        beta = zeros(p,1);
        beta(E) = pinv(X_E)*[Y;temp] - lambda*((X_E'*X_E)\Z_E);  
        yfit = X_withnew*beta;
        Resid = abs(yfit - [Y;temp]);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=1-alpha
            yconfupper = temp;
            yconf = [yconf temp];
        end
        temp = temp + stepsize;
    end
    if yconfupper == yconfupperold
        break;
    end    
    iterr = iterr + 1;
    fprintf('\t\tSupport Iteration Right:%d\n',iterr)
    yconfupperold = yconfupper;
    modelsize = modelsize+length(E);
end
modelsize = modelsize/(iterl+iterr+1);
supportcoverage = 1;
end
