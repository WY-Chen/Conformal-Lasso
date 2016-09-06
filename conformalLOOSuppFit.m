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
function [yconf,modelsize,supportcounter] = conformalLOOSuppFit(X,Y,xnew,alpha,ytrial,lambdain,initn)
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
betaN = glmnetCoef(glmnet(X,Y,[],options));
betaN=betaN(2:p+1);
ypred=xnew*betaN;
% message = sprintf('\tPrediction point is %2.2f', ypred);
% disp(message);
[maxOutErrNval,~] = max((X*betaN - Y).^2);
ytrial = ytrial(ytrial> ypred-sqrt(maxOutErrNval)...
    & ytrial < ypred+sqrt(maxOutErrNval));
n = length(ytrial); % new truncated trial set. 

%% Fit the trial set
stepsize = ytrial(2)-ytrial(1);

% Initial and Stopping range
yinit = min(ytrial);
yterm = max(ytrial);

% Fit initial Lasso
% Initiation
outlier = randi([1,m]);

% C-steps
outlierOld = outlier;
ccount=0;
Y_withnew=[Y;ytrial(1)];
while 1
    selection = setxor(1:m+1,outlier);
    beta = glmnetCoef(glmnet(X_withnew(selection,:),Y_withnew(selection),...
        [],options));
    beta = beta(2:p+1);
    [~,outlier]=max((X_withnew*beta-Y_withnew).^2);
    if outlier == outlierOld || ccount>20
        break
    end
    outlierOld = outlier;
    ccount=ccount+1;
end
selection = setxor(1:m+1,outlier);
E = find(beta);
Z = sign(beta);
X_E = X_withnew(selection,E);
% accelerate computation
pinvxe=pinv(X_E);
betalast = pinvxe(:,end);
betaincrement = zeros(p,1);
betaincrement(E) = betalast;
yfitincrement = X_withnew*betaincrement;
yfit = X_withnew*beta;
A = X_withnew(selection,:)'*([Y(selection(1:m-1));0]-yfit(selection));
left=xnew'-X_withnew(selection,:)'*yfitincrement(selection);
rightplus=lambdain-A-xnew'*yinit;
rightminus=-lambdain-A-xnew'*yinit;

supportcounter = 1;
ysmax=yinit;
yconf = []; modelsize=0;outlierFlag=0;

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
        [~,fitoutind]=max(Resid);
            
        % check if the selection is same
        if fitoutind == outlier
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconf = [yconf y];
            end
        else
            % if not the same outlier, the computation is invalid
            % redo this point in mode 1
            outlierFlag = 1;
            break
        end
    end
    % end of solving
    while 1
        if outlierFlag == 1
            outlier = randi([1,m]);
            
            % C-steps
            outlierOld = outlier;
            ccount=0;
            while 1
                selection = setxor(1:m+1,outlier);
                beta = glmnetCoef(glmnet(X_withnew(selection,:),Y_withnew(selection),...
                    [],options));
                beta = beta(2:p+1);
                [~,outlier]=max((X_withnew*beta-Y_withnew).^2);
                if outlier == outlierOld || ccount>20
                    break
                end
                outlierOld = outlier;
                ccount=ccount+1;
            end
            selection = setxor(1:m+1,outlier);
            E = find(beta);
            Z = sign(beta);
            Z_E=Z(E);
            X_E = X_withnew(selection,E);
            pinvxe=pinv(X_E);
            betalast = pinvxe(:,end);
            betaincrement = zeros(p,1);
            betaincrement(E) = betalast;
            yfitincrement = X_withnew*betaincrement;
            outlierFlag = 0;
            yfit = X_withnew*beta;
            break;
        else
            if ismember(stepind,E)
                E = E(E~=stepind);
                Z(stepind)=0;
                Z_E = Z(E);
            else
                E = sort([E;stepind]);
                Z(stepind)=stepsign;
                Z_E=Z(E);
            end
            selection = setxor(1:m+1,outlier);
            X_E = X_withnew(selection,E);
            xesquareinv = (X_E'*X_E)\eye(length(E));
            pinvxe=pinv(X_E);
            betalast = pinvxe(:,end);
            betaincrement = zeros(p,1);
            betaincrement(E) = betalast;
            yfitincrement = X_withnew*betaincrement;
            beta = zeros(p,1);
            beta(E) = pinvxe*[Y(selection(1:m-1));ysmax] - lambdain*xesquareinv*Z_E;
            yfit = X_withnew*beta;
            [~,fitoutind] = max(abs([Y;ysmax]-yfit));
            if fitoutind == outlier
                break;
            else
                outlierFlag = 1;
            end
        end
    end
    
    A = X_withnew(selection,:)'*([Y(selection(1:m-1));0]-yfit(selection));
    left=xnew'-X_withnew(selection,:)'*yfitincrement(selection);
    rightplus=lambdain-A-xnew'*ysmax;
    rightminus=-lambdain-A-xnew'*ysmax;
    
    supportcounter = supportcounter+1;
    if ysmax > yterm
        break
    end
end           
modelsize = modelsize/n;
        