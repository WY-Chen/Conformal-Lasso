%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new pair is outlier in model, if yes, use fit of known data
% Then:
% Check if the new pair is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize] = conformalLTSLassoAllSupp(X,Y,xnew,alpha,ytrial)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    a set of value to test

% prepare for fitting
addpath(genpath(pwd));
X_withnew = [X;xnew];
[m,p] = size(X);
n = length(ytrial);

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = 1.0;                % Lasso (no L2 norm penalty)
options.thresh = 1E-12;
Linoptions = optimset('Display','off');

% Build a model when new pair is outlier with only the known data
[betaN,~,~,~,H] = LTSlassoSupport(X,Y,xnew);
h=length(H);
[Rval,~] = sort((Y - X*betaN).^2);
bounds = [sqrt(Rval(h+1))+xnew*betaN,-sqrt(Rval(h+1))+xnew*betaN];
outmin = min(bounds);
outmax = max(bounds);
fprintf('\tPrediction point is %2.2f\n', xnew*betaN)


modelsizes = zeros(1,n);
compcase = 1; suppmax=-inf;
supportcounter = 0;
yconfidx = [];

wb = waitbar(0,'Please wait...');
i=1;
for i=1:n
    y = ytrial(i);
    if y<=outmin || y>=outmax
        continue;
    elseif y<=suppmax && y>=suppmin
        compcase=2;
    end
    switch compcase
        case 1
            % Compute full LTS-lasso
            [beta,A,b,lambda,H]=LTSlassoSupport(X_withnew,[Y;y],xnew);
            h=length(H);
            supportcounter = supportcounter+1;
            % updata support 
            E = find(beta);
            Z = sign(beta);
            Z_E = Z(E);
            [Rval,~] = sort((Y - X*beta).^2);
            bounds = [sqrt(Rval(h+1))+xnew*beta,-sqrt(Rval(h+1))+xnew*beta];
            outmin = min(bounds);
            outmax = max(bounds);
            if H(end)~=m+1
                continue
            end
            [suppmin,suppmax]=solveInt(A,b,[Y(H(1:(h-1)));y]);
            if suppmin>=suppmax
                fprintf('WARNING: bad support\n');
                continue;
            end
            disp([suppmin,suppmax])
            % conformal prediction
            Resid = abs(X_withnew*beta - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            if suppmin<suppmax
                compcase=2;
            end
        case 2
            % Compute LTS-lasso based on support
            beta = zeros(p,1);
            X_E = X_withnew(H,E);
            beta(E) = pinv(X_E)*[Y(H(1:(h-1)));y] - lambda*((X_E'*X_E)\Z_E);
            Resid = abs(X_withnew*beta - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            if ytrial(i+1)>suppmax
                compcase=1;
            end
    end
    modelsizes(i) = length(E);
    % waitbar
    waitbar(i/n,wb,sprintf('Number of Lasso support computed %d',supportcounter))
end
close(wb)
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