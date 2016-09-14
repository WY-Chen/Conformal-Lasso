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
Z_E=Z(E);
X_minusE = X_withnew(:,setxor(E,1:p));
X_E = X_withnew(:,E);
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
yold = yinit;
yfit = X_withnew*beta;
pinvxe=pinv(X_E);
betalast = pinvxe(:,end);
betaincrement = zeros(p,1);
betaincrement(E) = betalast;
yfitincrement = X_withnew*betaincrement;
ineqviolate = (b - A*[Y;yinit])./A(:,end);
supportmax = yinit+min(ineqviolate(ineqviolate>0));

supportcounter = 1;
yconfidx = []; compcase=2;
for i = 2:n
    y = ytrial(i);
    switch compcase
        case 2
            stepsize = ytrial(i)-ytrial(i-1);
            yfit = yfit + yfitincrement*stepsize;
            Resid = abs(yfit - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            % Change computation mode
            if ytrial(min(i+1,n))>supportmax
                compcase=1;
            end
        case 1
            yfit = X_withnew*beta + yfitincrement*(y-yold);
            xtres = X_withnew'*([Y;y]-yfit);
            ineqplus = find(xtres>lambdain);
            ineqminus = find(xtres<-lambdain);
            E = sort(unique([ineqplus; ineqminus]));
            Z = zeros(p,1);
            Z(ineqplus)=1;
            Z(ineqminus)=-1;
            Z_E=Z(E);
            X_minusE = X_withnew(:,setxor(E,1:p));
            X_E = X_withnew(:,E);
            yold = y;
            
            xesquareinv = (X_E'*X_E)\eye(length(E));
            pinvxe=pinv(X_E);
            betalast = pinvxe(:,end);
            betaincrement = zeros(p,1);
            betaincrement(E) = betalast;
            yfitincrement = X_withnew*betaincrement;
            beta = zeros(p,1);
            beta(E) = pinvxe*[Y;y] - xesquareinv*Z_E*lambdain;
            fprintf(2,'Refit: %d\n',length(E));
            % Robooting
            ineqviolated = find(abs(X_withnew'*([Y;y]-yfit))>=lambdain);
            if ~isequal(ineqviolated, E)
                beta = glmnetCoef(glmnet(X_withnew,[Y;y],[],options));
                beta=beta(2:p+1);
                E = find(beta);
                Z = sign(beta);
                Z_E=Z(E);
                X_minusE = X_withnew(:,setxor(E,1:p));
                X_E = X_withnew(:,E);
                fprintf(2,'Reboot: %d\n',length(E));
                % accelerate computation
                xesquareinv = (X_E'*X_E)\eye(length(E));
                pinvxe=pinv(X_E);
                betalast = pinvxe(:,end);
                betaincrement = zeros(p,1);
                betaincrement(E) = betalast;
                yfitincrement = X_withnew*betaincrement;
            end
            
            % accelerate the computation
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
            yfit = X_withnew*beta;
            % conformal
            Resid = abs(yfit - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            supportcounter = supportcounter+1;
            
            ineqviolate = (b - A*[Y;y])./A(:,end);
            supportmax = y+min(ineqviolate(ineqviolate>0));
            
            if supportmax >ytrial(min(n,i+1))
                compcase=2;
                pinvxe=pinv(X_E);
                betalast = pinvxe(:,end);
                betaincrement = zeros(p,1);
                betaincrement(E) = betalast;
                yfitincrement = X_withnew*betaincrement;
            end
    end
    modelsizes(i) = length(E);
%     % waitbar
%     waitbar(i/n,h,sprintf('Current model size %d. Number of Lasso support computed %d',...
%         length(E),supportcounter))
end
modelsize = mean(modelsizes);
yconf  = ytrial(yconfidx);
end