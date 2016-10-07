%% Conformal inference
% 1. Get rid of outliers: fit a lasso to the original N data. 
%   Rank the residues to compute a interval that the trial pair is included
%   in the selected set H. Discard the rest.
% 2. All Support method: similar to Lasso All support. 
%   Traverse the trial set, compute models by LTS-lasso on combined data: 
%       (a) If the new pair is within known polyhedron of support and signs,
%       apply the simplified computation on the selected set H; 
%       (b) if not, fit full LTS-lasso with C-Steps. 
%% Method
function [yconf,modelsize,sc] = conformalLTSLassoAllSupp(X,Y,xnew,alpha,ytrial,lambdain,tau)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    a set of value to test
% tau       proportion of error predetermined
% lambdain  initial lambda. Unlike others, this method does not have CV

%% Preparations for fitting
sc=0;
if nargin == 6
    tau=0.80;               % pre-determined proportion of good data
end
addpath(genpath(pwd));  % may use glmnet
X_withnew = [X;xnew];   % new combined data
[m,p] = size(X);        % X is m*p, Y is m*1
oldn = length(ytrial);  % total trial 


%% Fit the known data. 
% this is the condition of the new pair being outlier
[betaN,~,~,~,H] = LTSlassoSupport(X,Y,xnew,tau,lambdain);
hN=length(H);
[Rval,~] = sort((Y - X*betaN).^2);
bounds = [sqrt(Rval(hN))+xnew*betaN,-sqrt(Rval(hN))+xnew*betaN];
outminN = min(bounds);
outmaxN = max(bounds);
% message = sprintf('\tPrediction point is %2.2f', xnew*betaN);
% disp(message);
ytrial = ytrial(ytrial>outminN & ytrial<outmaxN);
n = length(ytrial); % new truncated trial set. 

%% Fit the trial set
modelsizes = zeros(1,n);
compcase = 1; outmin = -inf; outmax=inf;
supportcounter = 0; yconfidx = [];

wb = waitbar(0,'Please wait...');
for i=1:n
    waitbar((oldn-n+i)/oldn,wb,...
        sprintf('Current model size %d. Number of Lasso support computed %d'...
        ,modelsizes(max(i-1,1)),supportcounter))
    y = ytrial(i);
    % if new pair not in chosen model, use the competed model
    if y<=outmin || y>=outmax
        Resid = abs(X_withnew*betaN - [Y;y]);
        Pi_trial = sum(Resid<=Resid(end))/(m+1);
        if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
            yconfidx = [yconfidx i];
        end
        modelsizes(i)=length(find(betaN));
        continue;
    end
    switch compcase
        case 1
            % Compute full LTS-lasso
            [beta,A,b,lambda,H]=LTSlassoSupport(X_withnew,[Y;y],xnew,tau,lambdain);
            h=length(H);
            supportcounter = supportcounter+1;
            % updata support 
            E = find(beta);
            Z = sign(beta);
            Z_E = Z(E);
            modelsizes(i) = length(E);
            [Rval,~] = sort((Y - X*beta).^2);
            bounds = [sqrt(Rval(h))+xnew*beta,-sqrt(Rval(h))+xnew*beta];
            outmin = min(bounds);
            outmax = max(bounds);
            if H(end)~=m+1
                continue
            end
            % conformal prediction
            Resid = abs(X_withnew*beta - [Y;y]);
            Pi_trial = sum(Resid<=Resid(end))/(m+1);
            if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
                yconfidx = [yconfidx i];
            end
            if  all(A*[Y(H(1:(h-1)));ytrial(min(i+1,n))]-b<=0)
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
            if  ~all(A*[Y(H(1:(h-1)));ytrial(min(i+1,n))]-b<=0)
                compcase=1;
            end
            modelsizes(i) = length(E);
    end
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