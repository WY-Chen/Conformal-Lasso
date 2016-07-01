%% Run Testing
% Run testing on a setting with a method. 
%% Implementation
function runtest(setting,method,alpha,stepsize,nruns)
% Setting = 'A', 'B', 'C'.
%       A : Linear. X iid N(0,1). epsilon iid N(0,1). 
%       B : B-spline. X iid N(0,1). epsilon iid t(2).
%       C : Linear. X highly correlated. epsilon iid t(2).

% Method = 
%       ALL                  : run lasso for all. VERY SLOW
%       []  or LassoOneSupp  : Use one support, run Lassp
%       LassoAllSupp         : traverse all point in trial set, compute full
%                           lasso if not in known support, and give such
%                           support. Use subgradient method if in known
%                           support. 
%       ENOneSupp            : One Support method with elastic net 
%       LTSOneSupp           : One Support method with LTS lasso
%       LTSAllSupp           : All Support method with LTS lasso

% alpha = level of confidence

% stepsize = stepsize in searching

% nruns = total number of testing runs

% Default:
if ~exist('alpha','var')
    alpha = 0.05;
end
if ~exist('stepsize','var')
    stepsize = 0.01;
end
if ~exist('setting','var')
    setting = 'A';
end
if ~exist('nruns','var')
    nruns = 10;
end

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
end

% Testing
fprintf('TESTING SETTING %s, METHOD %s.\n',setting,method);
        
coverage = zeros(nruns,1);
for i=1:nruns
    fprintf('TESTING=== run %d/%d.\n',i,nruns);
    
    % Get testing data
    [X,Y,xnew,y] = getSetting(setting);
    
    % Get additional parameters to pass to method
    option = [min(Y):stepsize:max(Y)];    
    % run method
    [yconf,modelsize] = mtd(X,Y,xnew,alpha,option);
    if isempty(yconf)
        fprintf('WARNING: no valid point returned.\n')
        continue
    end
    coverage(i) = sum((min(yconf)<y)&(y<max(yconf)))/10000;

    % format print
    fprintf('\tAverage model size =%.1f\n',modelsize);
    fprintf('\tInterval [%f, %f].\n',min(yconf),max(yconf));
    fprintf('\tCoverage is %f\n',coverage(i));
end
fprintf('%d-fold average coverage is %f\n', nruns, mean(coverage))






