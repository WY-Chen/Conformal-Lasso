%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new point is in known support, if yes, subgradient method
% if no, run full Elastic net
%% Method
function [yconf,modelsize] = conformalLassoAllSupp(X,Y,xnew,alpha,ytrial)
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
options.alpha = 0.8;                % Elastic net
options.thresh = 1E-12;
Linoptions = optimset('Display','off');

% Build confidence interval
yconfidx = [];
[beta,A,b,lambda] = ENSupport(X_withnew,[Y;ytrial(1)],X_withnew,0.8);
E = find(beta);
Z = sign(beta);
Z_E = Z(E);
supportcounter = 1;
fprintf('\tPrediction point is %2.2f\n', xnew*beta)
supportmax = linprog(-1,A(:,m+1),b-A(:,1:m)*Y,[],[],[],[],[],Linoptions);
supportmin = linprog(1,A(:,m+1),b-A(:,1:m)*Y,[],[],[],[],[],Linoptions);

h = waitbar(0,'Please wait...');
for i = 1:n
    y = ytrial(i);
    if supportmin<= y && supportmax >=y
        beta = zeros(p,1);
        X_E = X_withnew(:,E);
        beta(E) = (X_E'*X_E+eye(length(E))*0.8)\X_E'*[Y;y]...
            - 0.8*lambda*((X_E'*X_E+eye(length(E))*0.8)\Z_E);  
        yfit = X_withnew*beta;
        Resid = abs(yfit - [Y;y]);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
            yconfidx = [yconfidx i];
        end
    else 
        [beta,A,b,lambda] = ENSupport(X_withnew,[Y;y],X_withnew,0.8);
        E = find(beta);
        Z = sign(beta);
        Z_E = Z(E);
        yfit = X_withnew*beta;
        Resid = abs(yfit - [Y;y]);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
            yconfidx = [yconfidx i];
        end
        supportcounter = supportcounter+1;
        supportmax = linprog(-1,A(:,m+1),b-A(:,1:m)*Y,[],[],[],[],[],Linoptions);
        supportmin = linprog(1,A(:,m+1),b-A(:,1:m)*Y,[],[],[],[],[],Linoptions);
    end   
    modelsizes(i) = length(E);
    % waitbar
    waitbar(i/n,h,sprintf('Number of Lasso support computed %d',supportcounter))
end
close(h)
modelsize = mean(modelsizes);
yconf  = ytrial(yconfidx);

% Plots
plotFlag=1;  % change to 0 to turn off
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