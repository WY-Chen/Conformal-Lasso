%% Run Testing
% Run testing on a setting with a method. 
%% Implementation
function runtest(setting,method,alpha,stepsize,nruns)
% Setting = 'A', 'B', 'C'.
%       A : Linear. X iid N(0,1). epsilon iid N(0,1). 
%       B : B-spline. X iid N(0,1). epsilon iid t(2).
%       C : Linear. X highly correlated. epsilon iid t(2).

% Method = 
%       ALL             : run lasso for all. VERY SLOW
%       []  or oneSupp  : brutal force search from lowest Y value to highest
%       predOneSupp     : traverse the support from prediction
%       predMultSupp    : traverse the support from prediction, then search
%                           support of lasso fitting known data and new trial 
%                           until no more trials are valid.
%       AllSupp         : traverse all point in trial set, compute full
%                           lasso if not in known support, and give such
%                           support. Use subgradient method if in known
%                           support. 

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
if ~exist('method')  | isequal(method,'oneSupp')
    mtd = @conformalLassoOneSupport;
    method = 'oneSupp';
elseif isequal(method,'predOneSupp')
    mtd = @conformalLassopredOneSupp;
elseif isequal(method,'predMultSupp')
    mtd = @conformalLassopredMultSupp;
elseif isequal(method,'ALL')
    mtd = @conformalLasso;
elseif isequal(method,'AllSupp')
    mtd = @conformalLassoAllSupp;
end

% Testing
fprintf('TESTING SETTING %s, METHOD %s.\n',setting,method);
        
coverage = zeros(nruns,1);
for i=1:nruns
    fprintf('TESTING=== run %d/%d.\n',i,nruns);
    
    % Get testing data
    [X,Y,xnew,y] = getSetting(setting);
    
    % Get additional parameters to pass to method
    if isequal(method,'oneSupp')| isequal(method,'ALL') |isequal(method,'AllSupp')
        option = [min(Y):stepsize:max(Y)];
    elseif isequal(method,'predOneSupp') | isequal(method,'predMultSupp') 
        option = stepsize;
    end
    
    % run method
    [yconf,supportcoverage,modelsize] = mtd(X,Y,xnew,alpha,option);
    coverage(i) = sum((min(yconf)<y)&(y<max(yconf)))/10000;

    % format print
    if isequal(method,'oneSupp')
        fprintf('\tModel size =%.1f\n',modelsize);
        fprintf('\t%.2f%% trials in support.\n',supportcoverage*100);
    else
        fprintf('\tAverage model size =%.1f\n',modelsize);
    end
    fprintf('\tInterval [%f, %f].\n',min(yconf),max(yconf));
    fprintf('\tCoverage is %f\n',coverage(i));
end
fprintf('%d-fold average coverage is %f\n', nruns, mean(coverage))






