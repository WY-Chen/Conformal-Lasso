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
h = 199;
H=randsample(1:200,h);
% Testing
        
coverage1 = zeros(nruns,1);
conflen1 = zeros(nruns,1);
coverage2 = zeros(nruns,1);
conflen2 = zeros(nruns,1);
% coverage3 = zeros(nruns,1);
% conflen3 = zeros(nruns,1);
coverage4 = zeros(nruns,1);
conflen4 = zeros(nruns,1);
time1=0;time2=0;time3=0;time4=0;
for i=1:nruns
    fprintf('TESTING=== run %d/%d.\n',i,nruns);
    
    % Get testing data
    [X,Y,xnew,y] = getSetting(setting);
    ytrial = [min(Y):stepsize:max(Y)];   
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

    % run method
    tic;
    [yconf1,modelsize1] = conformalLasso(X,Y,xnew,alpha,ytrial,lambda);
    t1=toc;time1=time1+t1;tic;
    [yconf2,modelsize2] = conformalLassoAllSupp(X,Y,xnew,alpha,ytrial,lambda);
    t2=toc;time2=time2+t2;tic;
%     [yconf3,modelsize3] = conformalLTSLassoAllSupp(X,Y,xnew,alpha,ytrial,lambda);
%     t3=toc;time3=time1+t3;tic;
    [yconf4,modelsize4] = conformalLOO(X,Y,xnew,alpha,ytrial,lambda);
    t4=toc;time4=time4+t4;
    coverage1(i) = sum((min(yconf1)<y)&(y<max(yconf1)))/10000;
    coverage2(i) = sum((min(yconf2)<y)&(y<max(yconf2)))/10000;
%     coverage3(i) = sum((min(yconf3)<y)&(y<max(yconf3)))/10000;
    coverage4(i) = sum((min(yconf4)<y)&(y<max(yconf4)))/10000;
    conflen1(i) = max(yconf1)-min(yconf1);
    conflen2(i) = max(yconf2)-min(yconf2);
%     conflen3(i) = max(yconf3)-min(yconf3);
    conflen4(i) = max(yconf4)-min(yconf4);

    % format print
    fprintf('\t\t\t\tALL\t\tLassoAllSupp\t\t\tLOO\n');
    fprintf('\tModelsize \t%.1f\t\t\t\t%.1f\t\t\t\t%.1f\n',modelsize1,modelsize2,modelsize4);
    fprintf('\tInterval \t[%.3f,%.3f]  [%.3f,%.3f] [%.3f,%.3f].\n',...
        min(yconf1),max(yconf1),min(yconf2),max(yconf2),min(yconf4),max(yconf4));
    fprintf('\tCoverage \t%.3f\t\t\t%.3f\t\t\t%.3f,\n',coverage1(i),coverage2(i),coverage4(i));
    fprintf('\tTime \t\t%.3f\t\t\t%.3f\t\t\t%.3f,\n',t1,t2,t4);
end
fprintf('%d-fold average coverage is %.3f, %.3f, %.3f\n', nruns, mean(coverage1),mean(coverage2),mean(coverage4))
fprintf('Average inverval length is %.1f, %.1f, %.1f\n',mean(conflen1),mean(conflen2),mean(conflen4));
fprintf('Average time is %.3f, %.3f, %.3f\n',time1/nruns,time2/nruns,time4/nruns);






