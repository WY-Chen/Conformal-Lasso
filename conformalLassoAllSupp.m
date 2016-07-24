%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new point is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize,supportcounter] = conformalLassoAllSupp(X,Y,xnew,alpha,ytrial,lambdain)
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
yconfidx = []; beta = zeros(p,1);
supportcounter = 0;

h = waitbar(0,'Please wait...');
compcase=1;
for i = 1:n
    y = ytrial(i);
    switch compcase
        case 2
            stepsize = ytrial(i)-ytrial(i-1);
            yfit = yfit + yfitincrement*stepsize;
            yfit = X_withnew*beta;
            Resid = abs(yfit - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            % Change computation mode
            if supportmin>= ytrial(min(i+1,n)) | ytrial(min(i+1,n))>=supportmax
                compcase=1;
            end
        case 1
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
            yfit = X_withnew*beta;
            % conformal
            Resid = abs(yfit - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            supportcounter = supportcounter+1;
            [supportmin,supportmax] = solveInt(A,b,Y);
            if supportmin< ytrial(min(n,i+1)) & supportmax >ytrial(min(n,i+1))
                compcase=2;
                pinvxe=pinv(X_E);
                betalast = pinvxe(:,end);
                betaincrement = zeros(p,1);
                betaincrement(E) = betalast;
                yfitincrement = X_withnew*betaincrement;
            end
    end
    modelsizes(i) = length(E);
    % waitbar
    waitbar(i/n,h,sprintf('Current model size %d. Number of Lasso support computed %d',...
        length(E),supportcounter))
end
close(h)
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