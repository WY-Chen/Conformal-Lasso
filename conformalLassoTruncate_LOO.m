%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new point is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize,supportcounter,triallen] = conformalLassoTruncate_LOO(X,Y,xnew,alpha,ytrial,lambdain)
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

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = 1.0;                % Lasso (no L2 norm penalty)
options.thresh = 1E-12;
options.nlambda = 1;
options.lambda = lambdain/m;

%% Fit the known data. 
% this is the condition of the new pair being outlier
% use this condition to truncate the trial set
betaN = glmnetCoef(glmnet(X,Y,[],options));
betaN=betaN(2:p+1);
ypred=xnew*betaN;
% message = sprintf('\tPrediction point is %2.2f', ypred);
% disp(message);
[maxOutErrNval,~] = max(abs(X*betaN - Y));
ytrial = ytrial(ytrial> ypred-maxOutErrNval...
    & ytrial < ypred+maxOutErrNval);
n = length(ytrial); % new truncated trial set. 
triallen = max(ytrial)-min(ytrial);
modelsizes = zeros(1,n);

%% Initialization with the first point
yconfidx = []; beta = zeros(p,1);
supportcounter = 1;

E = find(betaN);Z=sign(beta);Z_E=Z(E);
X_E = X_withnew(:,E);
X_minusE = X_withnew(:,setxor(E,1:p));
xesquareinv = (X_E'*X_E)\eye(length(E));
P_E = X_E*xesquareinv*X_E';
a0=X_minusE'*(eye(m+1)-P_E)./lambdain;
temp = X_minusE'*pinv(X_E')*Z_E;
% calculate the inequalities for fitting.
A = [a0;
    -a0;
    -diag(Z_E)*xesquareinv*X_E'];
b = [ones(p-length(E),1)-temp;
    ones(p-length(E),1)+temp;
    -lambdain*diag(Z_E)*xesquareinv*Z_E];
if all(A*[Y;ytrial(1)]<b)
    compcase=2;
    yfit = X_withnew*beta;
    pinvxe=pinv(X_E);
    betalast = pinvxe(:,end);
    betaincrement = zeros(p,1);
    betaincrement(E) = betalast;
    yfitincrement = X_withnew*betaincrement;
    ineqviolate = (b - A*[Y;ypred])./A(:,end);
    supportmax = ypred+min(ineqviolate(ineqviolate>0));
else
    compcase=1;
end

% h = waitbar(0,'Please wait...');
changeind = -inf;
for i = 1:n
    y = ytrial(i);
    switch compcase
        case 2
            yfit = yfit + yfitincrement*stepsize;
            Resid = abs(yfit - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            % Change computation mode
            if ytrial(min(i+1,n))>supportmax
                changeind = find(A*[Y;ytrial(min(i+1,n))]>=b);
                if isempty(changeind) | length(changeind)>1
                    changeind=-inf;
                end
                compcase=1;
            end
        case 1
            if changeind < 0
                beta = glmnetCoef(glmnet(X_withnew,[Y;y],[],options));
                beta=beta(2:p+1);
                E = find(beta);
                Z = sign(beta);
                Z_E = Z(E);
                X_E = X_withnew(:,E);
                X_minusE = X_withnew(:,setxor(E,1:p));
                pinvxe = pinv(X_E);
                xesquareinv = (X_E'*X_E)\eye(length(E));
            elseif changeind > 2*(p-length(E))
                changeind = changeind-2*(p-length(E));
                changeind = E(changeind);
                E = E(E~=changeind);
                Z(changeind)=0;
                Z_E = Z(E);
                X_E = X_withnew(:,E);
                X_minusE = X_withnew(:,setxor(E,1:p));
                % accelerate the computation
                pinvxe = pinv(X_E);
                xesquareinv = (X_E'*X_E)\eye(length(E));
                beta = zeros(p,1);
                if ~isempty(E)
                    beta(E) = pinvxe*[Y;y] - xesquareinv*Z_E*lambdain;
                end
%                 fprintf('r%d\n',changeind);
            elseif changeind > p-length(E)
                changeind = changeind - (p-length(E));
                k = changeind;
                for j=E'
                    if j<=k
                        k=k+1;
                    end
                end
                changeind = k;
                E = unique(sort([E;k]));
                Z(k)=-1;
                Z_E=Z(E);
                X_E = X_withnew(:,E);
                X_minusE = X_withnew(:,setxor(E,1:p));
                % accelerate the computation
                pinvxe = pinv(X_E);
                xesquareinv = (X_E'*X_E)\eye(length(E));
                beta = zeros(p,1);
                beta(E) = pinvxe*[Y;y] - xesquareinv*Z_E*lambdain;
%                 fprintf('-%d\n',changeind);
            elseif 0<changeind
                k = changeind;
                for j=E'
                    if j<=k
                        k=k+1;
                    end
                end
                changeind = k;
                E = unique(sort([E;k]));
                Z(k)=1;
                Z_E=Z(E);
                X_E = X_withnew(:,E);
                X_minusE = X_withnew(:,setxor(E,1:p));
                % accelerate the computation
                pinvxe = pinv(X_E);
                xesquareinv = (X_E'*X_E)\eye(length(E));
                beta = zeros(p,1);
                beta(E) = pinvxe*[Y;y] - xesquareinv*Z_E*lambdain;
%                 fprintf('+%d\n',changeind);
            else
                fprintf(2,'ERROR: index not found\n');
            end
                
            

            P_E = X_E*xesquareinv*X_E';
            if isempty(E)
                temp=0;
            else
                temp = X_minusE'*pinv(X_E')*Z_E;
            end
                
            a0=X_minusE'*(eye(m+1)-P_E)./lambdain;
            % calculate the inequalities for fitting.
            A = [a0;
                -a0;
                -diag(Z_E)*xesquareinv*X_E'];
            b = [ones(p-length(E),1)-temp;
                ones(p-length(E),1)+temp;
                -lambdain*diag(Z_E)*xesquareinv*Z_E];
            yfit = X_withnew*beta;
            ineqviolate = (b - A*[Y;y])./A(:,end);
            supportmax = y+min(ineqviolate(ineqviolate>0));

            % conformal
            Resid = abs(yfit - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            supportcounter = supportcounter+1;

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
% close(h)
modelsize = mean(modelsizes);
yconf  = ytrial(yconfidx);

end