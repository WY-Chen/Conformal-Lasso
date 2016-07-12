%% Run Testing
% Run testing on a setting with a method. 
%% Implementation
function runtest(setting,method,alpha,stepsize,nruns,CV)
% Setting = 'A', 'B', 'C'.
%       A : Linear. X iid N(0,1). epsilon iid N(0,1). 
%       B : B-spline. X iid N(0,1). epsilon iid t(2).
%       C : Linear. X highly correlated. epsilon iid t(2).

% Method = 
%       ALL                  : run lasso for all. VERY SLOW
%       LassoAllSupp         : traverse all point in trial set, compute full
%                           lasso if not in known support, and give such
%                           support. Use subgradient method if in known
%                           support. 
%       LTSAllSupp           : All Support method with LTS lasso
%       LOO                  : leave one out version of LTS

% BAD METHODS: just for comparison
%       LassoOneSupp  : Use one support, run Lassp
%       ENOneSupp            : One Support method with elastic net 
%       LTSOneSupp           : One Support method with LTS lasso

% CV = off by default. Turn on for comparison

% alpha = level of confidence

% stepsize = stepsize in searching

% nruns = total number of testing runs

% Default:
if ~exist('alpha','var')
    alpha = 0.05;
end
if ~exist('stepsize','var')
    stepsize = 0.1;
end
if ~exist('setting','var')
    setting = 'A';
end
if ~exist('nruns','var')
    nruns = 10;
end
if ~exist('CV','var')
    CV = 'off';
end

H=1:200;h=200;
% Formatting methods
if ~exist('method')  | isequal(method,'LassoOneSupp')
    mtd = @conformalLassoOneSupp;
    method = 'LassoOneSupp';
elseif isequal(method,'ALL')
    mtd = @conformalLasso;
elseif isequal(method,'LassoAllSupp')
    mtd = @conformalLassoAllSupp;
elseif isequal(method,'ENOneSupp')
    mtd = @conformalENOneSupp;
elseif isequal(method,'LTSOneSupp')
    mtd = @conformalLTSLassoOneSupp;
elseif isequal(method,'LTSAllSupp')
    mtd = @conformalLTSLassoAllSupp;
    h = 190;
    H=randsample(1:200,h);
elseif isequal(method,'LOO')
    mtd = @conformalLOO;
    h = 199;
    H=randsample(1:200,h);
end

% Testing
fprintf('TESTING SETTING %s, METHOD %s.\n',setting,method);
        
coverage = zeros(nruns,1);
conflen = zeros(nruns,1);
time = 0;
for i=1:nruns
    fprintf('TESTING=== run %d/%d.\n',i,nruns);
    
    % Get testing data
    [X,Y,xnew,y] = getSetting(setting);
    
    % Get lambda from empirical expectation
    t=0;
    for j=1:100
        if setting=='A'
            epsilon = normrnd(0,1,[h,1]);
        else
            epsilon = trnd(2,[h,1]);
        end
        t=t+norm(X(H,:)'*epsilon,inf)*2;
    end
    lambda = t/100;
    
    % Get ytrial 
    ytrial = [min(Y):stepsize:max(Y)];    
    
    % run method
    tic;
    if isequal(CV,'CV')
        [yconf,modelsize] = mtd(X,Y,xnew,alpha,ytrial);
    else
        [yconf,modelsize] = mtd(X,Y,xnew,alpha,ytrial,lambda);
    end
    if isempty(yconf)
        fprintf('WARNING: no valid point returned.\n')
        continue
    end
    coverage(i) = sum((min(yconf)<y)&(y<max(yconf)))/10000;
    conflen(i) = max(yconf)-min(yconf);
    if coverage(i)<0.4
        save('broken_case.mat','X','Y','xnew');
    end
    t=toc; time=time+t;
    % format print
    fprintf('\tAverage model size =%.1f\n',modelsize);
    fprintf('\tInterval [%f, %f].\n',min(yconf),max(yconf));
    fprintf('\tCoverage is %f\n',coverage(i));
    fprintf('\tElapsed time is %.2f\n',t);
end
fprintf('%d-fold average coverage is %f\n', nruns, mean(coverage))
fprintf('Average inverval length =%.1f\n',mean(conflen));
fprintf('Average time =%.2f\n',time/nruns);






