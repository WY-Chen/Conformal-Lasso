%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new point is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize] = conformalLTSLassoAllSupp(X,Y,xnew,alpha,ytrial)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% stepsize  stepsize of searching for upper and lower bound of interval

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
Linoptions = optimset('Display','off');

% Build confidence interval
yconfidx = [];
[beta,A,b,lambda,H] = LTSlassoSupport(X_withnew,[Y;ytrial(1)],xnew);
h=length(H);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
supportcounter = 1;
fprintf('\tPrediction point is %2.2f\n', xnew*beta)
supportmax = linprog(-1,A(:,h+1),realmin+b-A(:,1:h)*Y(H),[],[],[],[],[],Linoptions);
supportmin = linprog(1,A(:,h+1),realmin+b-A(:,1:h)*Y(H),[],[],[],[],[],Linoptions);
supportmin = min(supportmin,supportmax);
supportmax = max(supportmin,supportmax);

wb = waitbar(0,'Please wait...');
i=1;
while i<=n
    y = ytrial(i);
    yfit = X_withnew*beta;
    Resid = ([Y;y] - yfit).^2;
    [Rval Rind] = sort(Resid);
    Hknew = Rind(1:h);
    Hknew = sort(Hknew);
    if ~ismember(m+1,Hknew)
%         Resid = abs(yfit - [Y;y]);
%         Pi_trial = sum(Resid<=Resid(end))/(m+1);
%         if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
%             yconfidx = [yconfidx i];
%         end
        i=i+1;
    elseif supportmin<= y && supportmax >=y
        beta = zeros(p,1);
        X_E = X_withnew([H;m+1],E);
        beta(E) = pinv(X_E)*[Y(H);y] - lambda*((X_E'*X_E)\Z_E);
        yfit = X_withnew*beta;
        Resid = abs(yfit - [Y;y]);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
            yconfidx = [yconfidx i];
        end
        i=i+1;
    else
        [beta,A,b,lambda,H] = LTSlassoSupport(X_withnew,[Y;ytrial(1)],xnew);
        E = find(beta);
        Z = sign(beta);
        Z_E = Z(E);
        supportmax = linprog(-1,A(:,h+1),realmin+b-A(:,1:h)*Y(H),[],[],[],[],[],Linoptions);
        supportmin = linprog(1,A(:,h+1),realmin+b-A(:,1:h)*Y(H),[],[],[],[],[],Linoptions);
        supportmin = min(supportmin,supportmax);
        supportmax = max(supportmin,supportmax);
        yfit = X_withnew*beta;
        Resid = abs(yfit - [Y;y]);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
            yconfidx = [yconfidx i];
        end
        supportcounter = supportcounter+1;
        i=i+1;
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