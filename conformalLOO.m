%% Conformal inference with LOO lasso
% 1. Get rid of outlier: fit lasso to known data, 
%   The prediction value plus/minus the largest residue is the interval 
%   for trial values that is excluded in the selected data set. Discard.
% 2. Fit lasso for each point: Start with mode 1, 
%   do the following two modes of computation:
%    1. Mode 1: apply initialization and C-steps, get a LTS-Lasso fit. 
%               Do conformal prediction. 
%               Then check if next trial value is within the polyhedron, 
%               if yes, switch to mode 2. If not, continue with mode 1. 
%    2. Mode 2: use the known support and signs to refit the data, 
%               rank the residues to check if the outlier is also the same.
%               (a) If yes, proceed with mode 2 on the next trial value 
%                       until the next one is not in known support. 
%               (b) If not, rerun with mode 1.
%% Method
function [yconf,modelsize,supportcounter] = conformalLOO(X,Y,xnew,alpha,ytrial,lambdain)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    a set of value to test
% lambdain  initial lambda. Unlike others, this method does not have CV
% initn     number of initializations. By default set to 0.  

%% Preparations for fitting
addpath(genpath(pwd));  % may use glmnet
X_withnew = [X;xnew];   % new combined data
[m,p] = size(X);        % X is m*p, Y is m*1
oldn = length(ytrial);  % total trial 

%% Tune GLMNET
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

%% Fit the trial set
i=1; compcase=1;yconfidx=[];
% h = waitbar(0,'Please wait...');
modelsizes = zeros(1,n); supportcounter=1;
while i<=n
    y=ytrial(i); Y_withnew = [Y;y];
    switch compcase
        case 1
            % recompute full lasso: C-steps
            
            % Initiation
            outlier = randi([1,m]);
            
            % C-steps
            outlierOld = outlier;
            ccount=0;
            selection = setxor(1:m+1,outlier);
            beta = glmnetCoef(glmnet(X_withnew(selection,:),Y_withnew(selection),...
                    [],options));
            beta=beta(2:p+1);
            while 1
                % given beta, choose H
                [~,outlier]=max(abs(X_withnew*beta-Y_withnew));
                if outlier == outlierOld || ccount>20
                    break
                end
%                 fprintf(2,'C-step %d outlier %d\n',ccount, outlier);
                selection = setxor(1:m+1,outlier);
                % given H, choose beta
                q = X_withnew(selection,:)'*...
                    (X_withnew(selection,:)*beta-Y_withnew(selection))./(m-1);
                % line search 
                ita = 1;
                snew = softthresh(ita*lambdain/(m-1),beta-ita*q);
%                 s = beta;
%                 sfit = X_withnew(selection,:)*s;
%                 snewfit = X_withnew(selection,:)*snew;
%                 normold = norm(sfit-Y_withnew(selection));
%                 while 0<normold-norm(snewfit-Y_withnew(selection))<=-ita*sum(q)./2
%                     ita = ita*.8;
%                     snew = softthresh(ita,beta-ita*q);
%                     snewfit = X_withnew(selection,:)*snew;
% %                     fprintf(2,'+');
%                 end
% %                 fprintf(2,'%.2f\n',ita);
                beta = snew;
                if outlier == m+1
                    outlier=outlierOld;
                    break;
                end
                outlierOld = outlier;
                ccount=ccount+1;
            end
            
            % re-fit (because above is inaccurate)
            if ccount>0
                selection = setxor(1:m+1,outlier);
                beta = glmnetCoef(glmnet(X_withnew(selection,:),Y_withnew(selection),...
                    [],options));
                beta=beta(2:p+1);
            end
            
            % Conformal
            yfit = X_withnew*beta;
            Resid = abs(yfit - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            
            
            % compute the sign/supports
            E = find(beta); Z = sign(beta); Z_E = Z(E);
            X_minusE = X_withnew(selection,setxor(E,1:p));
            X_E = X_withnew(selection,E);
            % accelerate computation
            xesquareinv = (X_E'*X_E)\eye(length(E));
            temp = X_minusE'*pinv(X_E')*Z_E;
            P_E = X_E*xesquareinv*X_E';
            a0=X_minusE'*(eye(m)-P_E)./lambdain;
            % calculate the inequalities for fitting.
            A = [a0;
                -a0;
                -diag(Z_E)*xesquareinv*X_E'];
            b = [ones(p-length(E),1)-temp;
                ones(p-length(E),1)+temp;
                -lambdain*diag(Z_E)*xesquareinv*Z_E];
            ineqviolate = (b - A*Y_withnew(selection))./A(:,end);
            supportmax = y+min(ineqviolate(ineqviolate>0));
            
            % Change computation mode
            if ytrial(min(i+1,n))<=supportmax
                compcase=2;
                % the following is to ease computation in mode 2
                pinvxe=pinv(X_E);
                betalast = pinvxe(:,end);
                betaincrement = zeros(p,1);
                betaincrement(E) = betalast;
                yfitincrement = X_withnew*betaincrement;
            end
            supportcounter = supportcounter+1;
        case 2
            % Fit the known support/sign
            stepsize = ytrial(i)-ytrial(i-1);
            yfit = yfit + yfitincrement*stepsize;
            Resid = abs(yfit - [Y;y]);
            [~,fitoutind]=max(Resid);
            
            % check if the selection is same
            if fitoutind == outlier
                Pi_trial = sum(Resid<=Resid(end))/(m+1);
                if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                    yconfidx = [yconfidx i];
                end
            else
                % if not the same outlier, the computation is invalid
                % redo this point in mode 1
                compcase=1;
                continue;
            end
            
            % Change computation mode
            if ytrial(min(i+1,n))>supportmax
                compcase=1;
            end
    end
%     waitbar((oldn-n+i)/oldn,h,...
%         sprintf('Current model size %d. Number of Lasso support computed %d',...
%         length(E),supportcounter))
    modelsizes(i)=length(E);
    i=i+1;
end
% close(h);
modelsize = mean(modelsizes);
yconf  = ytrial(yconfidx);
end