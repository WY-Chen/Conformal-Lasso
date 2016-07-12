%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new pair is outlier in model, if yes, use fit of known data
% Then:
% Check if the new pair is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize] = conformalLOO(X,Y,xnew,alpha,ytrial,lambdain)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    a set of value to test
% tau       proportion of error predetermined

% prepare for fitting
addpath(genpath(pwd));
X_withnew = [X;xnew];
[m,p] = size(X);

% initialize beta and E
betaN = lasso(X,Y,'Lambda',lambdain/m);
[maxOutErrNval,~] = max((X*betaN - Y).^2);
ytrial = ytrial(ytrial> xnew*betaN-sqrt(maxOutErrNval)...
    & ytrial < xnew*betaN+sqrt(maxOutErrNval));
n = length(ytrial);

i=1; compcase=1;yconfidx=[];
h = waitbar(0,'Please wait...');
modelsizes = zeros(1,n); supportcounter=0;
while i<=n
    y=ytrial(i); Y_withnew = [Y;y];
    switch compcase
        case 1
            % recompute full lasso: C-steps
            
            %initiation
            initOuts = zeros(1,10);
            for j=1:10
                init = randsample(1:m+1,3);
                initlambda = 2*norm((X_withnew(init,:)'*normrnd(0,1,[3,1])),inf);
                beta = lasso(X_withnew(init,:),Y_withnew(init),'Lambda',initlambda);
                [~,initOut]=max((X_withnew*beta-Y_withnew).^2);
                selection = setxor(1:m+1,initOut);
                beta = lasso(X_withnew(selection,:),Y_withnew(selection),'Lambda',lambdain/m);
                [~,initOut]=max((X_withnew*beta-Y_withnew).^2);
                initOuts(j) = initOut;
            end
            outlier = mode(initOuts);
            
            % C-steps
            outlierOld = outlier;
            ccount=0;
            while 1
                selection = setxor(1:m+1,outlier);
                beta = lasso(X_withnew(selection,:),Y_withnew(selection),...
                    'Lambda',lambdain/m,'Standardize',0,'RelTol',1E-8);
                [~,outlier]=max((X_withnew*beta-Y_withnew).^2);
                if outlier == outlierOld || ccount>20
                    break
                end
                outlierOld = outlier;
                ccount=ccount+1;
            end
            
            yfit = X_withnew*beta;
            Resid = abs(yfit - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            
            E = find(beta); Z = sign(beta); Z_E = Z(E);
            X_minusE = X_withnew(selection,setxor(E,1:p));
            X_E = X_withnew(selection,E);
            P_E = X_E*((X_E'*X_E)\X_E');
            % calculate the inequalities for fitting.
            A = [X_minusE'*(eye(m)-P_E)./lambdain;
                -X_minusE'*(eye(m)-P_E)./lambdain;
                -diag(Z_E)*((X_E'*X_E)\X_E')];
            b = [ones(p-length(E),1)-X_minusE'*pinv(X_E')*Z_E;
                ones(p-length(E),1)+X_minusE'*pinv(X_E')*Z_E;
                -lambdain*diag(Z_E)*((X_E'*X_E)\Z_E)];
            if max(A*[Y(selection(1:m-1));ytrial(min(i+1,n))]-b)<=0
                compcase=2;
            end
            supportcounter = supportcounter+1;
        case 2
            beta = zeros(p,1);
            X_E = X_withnew(selection,E);
            beta(E) = pinv(X_E)*Y_withnew(selection) - lambdain*((X_E'*X_E)\Z_E);
            yfit = X_withnew*beta;
            Resid = abs(yfit - [Y;y]);
            [~,fitoutind]=max(Resid);
            if fitoutind == outlier
                Pi_trial = sum(Resid<=Resid(end))/(m+1);
                if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                    yconfidx = [yconfidx i];
                end
            else
                compcase=1;
                continue;
            end
    end
    waitbar(i/n,h,sprintf('Current model size %d. Number of Lasso support computed %d',...
        length(E),supportcounter))
    modelsizes(i)=length(E);
    i=i+1;
end
close(h);
modelsize = mean(modelsizes);
yconf  = ytrial(yconfidx);
            
        
        