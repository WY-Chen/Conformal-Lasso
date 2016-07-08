%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new pair is outlier in model, if yes, use fit of known data
% Then:
% Check if the new pair is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize] = conformalLTSLassoAllSupp(X,Y,xnew,alpha,ytrial,lambdain)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    a set of value to test
% tau       proportion of error predetermined

tau=0.95;

% prepare for fitting
addpath(genpath(pwd));
X_withnew = [X;xnew];
[m,p] = size(X);
n = length(ytrial);

% Build a model when new pair is outlier with only the known data
[betaN,~,~,~,H] = LTSlassoSupport(X,Y,xnew,tau,lambdain);
hN=length(H);
[Rval,~] = sort((Y - X*betaN).^2);
bounds = [sqrt(Rval(hN))+xnew*betaN,-sqrt(Rval(hN))+xnew*betaN];
outminN = min(bounds);
outmaxN = max(bounds);
fprintf('\tPrediction point is %2.2f\n', xnew*betaN)


modelsizes = zeros(1,n);
compcase = 1; suppmax=-inf; outmin = -inf; outmax=inf;
supportcounter = 0; yconfidx = [];


wb = waitbar(0,'Please wait...');
for i=1:n
    waitbar(i/n,wb,...
        sprintf('Current model size %d. Number of Lasso support computed %d'...
        ,modelsizes(max(i-1,1)),supportcounter))
    y = ytrial(i);
    % if new pair not in chosen model, use the competed model
    if y<=outminN || y>=outmaxN || y<=outmin || y>=outmax
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
%             [suppmin,suppmax]=solveInt(A,b,Y(H(1:(h-1))));
%             if suppmin>=suppmax
%                 fprintf('WARNING: bad support %d size %d\n',...
%                     supportcounter, length(E));
%             end
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