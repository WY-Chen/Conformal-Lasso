%% Run Testing
% Run testing on a setting with a method. 
%% Implementation
function runtest(setting,method,alpha,stepsize,nruns)
% Setting = 'A', 'B', 'C'.
%       A : Linear. X iid N(0,1). epsilon iid N(0,1). 
%       B : B-spline. X iid N(0,1). epsilon iid t(2).
%       C : Linear. X highly correlated. epsilon iid t(2).

% Method = 
%       []  or BF   : brutal force search from lowest Y value to highest
%       predSupp    : treaverse the support from prediction

% alpha = level of confidence

% stepsize = stepsize in searching

% nruns = total number of runs

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


% Testing
if not(exist('method'))
    mtd = @conformalLassoWithSupport;
    method = 'BF';
elseif method == 'predSupp'
    mtd = @conformalLassoWithSupportSearch;
end
fprintf('TESTING SETTING %s, METHOD %s.\n',setting,method);
        
coverage = zeros(nruns,1);
for i=1:nruns
    [X,Y,xnew,y] = getSetting(setting);
    fprintf('TESTING=== run %d/%d.\n',i,nruns);
    if isequal(method,'BF')
        option = [min(Y):stepsize:max(Y)];
    elseif isequal(method,'predSupp')
        option = stepsize;
    end
    
    [yconf,supportcoverage,model] = mtd(X,Y,xnew,alpha,option);
    fprintf('\tModel size =%d\n',length(model))
    fprintf('\t%.2f%% trials in support.\n',supportcoverage*100);
    fprintf('\tInterval [%f, %f].\n',min(yconf),max(yconf));
    
    if supportcoverage==0
        continue
    end
    
    coverage(i) = sum((min(yconf)<y)&(y<max(yconf)))/10000;
    fprintf('\tCoverage is %f\n',coverage(i))
end
fprintf('10-fold average coverage is %f\n', mean(coverage))






