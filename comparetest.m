%% Run Testing
% Run testing on a setting with a method. 
%% Implementation
function comparetest(setting,alpha,stepsize,nruns)
% Setting = 'A', 'B', 'C'.
%       A : Linear. X iid N(0,1). epsilon iid N(0,1). 
%       B : B-spline. X iid N(0,1). epsilon iid t(2).
%       C : Linear. X highly correlated. epsilon iid t(2).

% Method = 
%       LassoAllSupp         : traverse all point in trial set, compute full
%                           lasso if not in known support, and give such
%                           support. Use subgradient method if in known
%                           support. 
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

% Testing
        
coverage1 = zeros(nruns,1);
conflen1 = zeros(nruns,1);
coverage2 = zeros(nruns,1);
conflen2 = zeros(nruns,1);
for i=1:nruns
    fprintf('TESTING=== run %d/%d.\n',i,nruns);
    
    % Get testing data
    [X,Y,xnew,y] = getSetting(setting);
    
    % Get additional parameters to pass to method
    option = [min(Y):stepsize:max(Y)];    
    % run method
    [yconf1,modelsize1] = conformalLassoAllSupp(X,Y,xnew,alpha,option);
    [yconf2,modelsize2] = conformalLTSLassoAllSupp(X,Y,xnew,alpha,option);
    coverage1(i) = sum((min(yconf1)<y)&(y<max(yconf1)))/10000;
    coverage2(i) = sum((min(yconf2)<y)&(y<max(yconf2)))/10000;
    conflen1(i) = max(yconf1)-min(yconf1);
    conflen2(i) = max(yconf2)-min(yconf2);

    % format print
    fprintf('\tAverage model size =%.1f and %.1f\n',modelsize1,modelsize2);
    fprintf('\tInterval [%f, %f].\n',min(yconf1),max(yconf1));
    fprintf('\tInterval [%f, %f].\n',min(yconf2),max(yconf2));
    fprintf('\tCoverage is %f and %f\n',coverage1(i),coverage2(i));
    fprintf('\tLength is %f and %f.\n',conflen1(i),conflen2(i));
end
fprintf('%d-fold average coverage is %f and %f\n', nruns, mean(coverage1),mean(coverage2))
fprintf('Average inverval length is %.1f and %.1f\n',mean(conflen1),mean(conflen2));






