%% Run Testing
% Run testing on a setting with a method. 
%% Implementation
function compareLOOtest(setting,tail,alpha,stepsize,nruns,filename)
% Setting = 'A', 'B', 'C'.
%       A : Linear. X iid N(0,1). epsilon iid N(0,1). 
%       B : B-spline. X iid N(0,1). epsilon iid t(2).
%       C : Linear. X highly correlated. epsilon iid t(2).

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
if ~exist('filename','var')
    fileID = 1;
else
    folder = fullfile(pwd, '\Outputs');
    filename = sprintf('Setting%s%s_%dIterations.txt',setting,tail,nruns);
    fileID = fopen(fullfile(folder, filename),'w');
end
% Testing
        
fprintf(fileID,'TESTING SETTING %s.\n',setting);
coverage1 = zeros(nruns,1);
coverage2 = zeros(nruns,1);
coverage3 = zeros(nruns,1);
coverage4 = zeros(nruns,1);
conflen1 = zeros(nruns,1);
conflen2 = zeros(nruns,1);
conflen3 = zeros(nruns,1);
conflen4 = zeros(nruns,1);
time1=0;time2=0;time3=0;time4=0;
totalsp1=0;totalsp2=0;totalsp3=0;totalsp4=0;
for i=1:nruns
    fprintf(fileID,'TESTING=== run %d/%d.\n',i,nruns);
    fprintf(2,'TESTING=== run %d/%d.\n',i,nruns);
    
    % Get testing data
    [X,Y,xnew,y] = getSetting(setting,tail);
    X_withnew = [X;xnew];
    ytrial = min(Y):stepsize:max(Y);   
    % Get lambda from empirical expectation
    t=0;
    for j=1:100
        if strcmp(tail,'norm')
            epsilon = normrnd(0,1,[201,1]);
        else
            epsilon = trnd(2,[201,1]);
        end
        t=t+norm(X_withnew'*epsilon,inf)*2;
    end
    lambda = t/100;
    if setting=='B'
        if strcmp(tail,'norm')
            lambda=628;
        else
            lambda=1016;
        end
        range = max(Y)-min(Y);
        ytrial = (min(Y)-range/2):stepsize:(max(Y)+range/2); 
    end


    % run method
    tic;
    [yconf1,modelsize1,sc1] = conformalLassoAllSupp(X,Y,xnew,alpha,ytrial,lambda);
    t1=toc;time1=time1+t1;tic;
    [yconf2,modelsize2,sc2] = conformalLassoSuppFit(X,Y,xnew,alpha,ytrial,lambda);
    t2=toc;time2=time2+t2;tic;
    [yconf3,modelsize3,sc3] = conformalLOO(X,Y,xnew,alpha,ytrial,lambda);
    t3=toc;time3=time3+t3;
    [yconf4,modelsize4,sc4] = conformalLOOSuppFit(X,Y,xnew,alpha,ytrial,lambda);
    t4=toc;time4=time4+t4;
    totalsp1 = totalsp1+sc1;
    totalsp2 = totalsp2+sc2;
    totalsp3 = totalsp3+sc3;
    totalsp4 = totalsp4+sc4;
    if isempty(yconf1)
        yconf1=ytrial;
    end
    if isempty(yconf2)
        yconf2=ytrial;
    end
    if isempty(yconf3)
        yconf3=ytrial;
    end
    coverage1(i) = sum((min(yconf1)<y)&(y<max(yconf1)))/10000;
    coverage2(i) = sum((min(yconf2)<y)&(y<max(yconf2)))/10000;
    coverage3(i) = sum((min(yconf3)<y)&(y<max(yconf3)))/10000;
    coverage4(i) = sum((min(yconf4)<y)&(y<max(yconf4)))/10000;
    conflen1(i) = max(yconf1)-min(yconf1);
    conflen2(i) = max(yconf2)-min(yconf2);
    conflen3(i) = max(yconf3)-min(yconf3);
    conflen4(i) = max(yconf4)-min(yconf4);
    % format print
    fprintf(fileID,'\t\t\t\tLassoAllSupp\tLassoSuppFit\t\tLOO\t\tLOOSuppfit\n');
    fprintf(fileID,'\tModelsize \t%.1f\t\t\t\t%.1f\t\t\t\t%.1f\t\t\t\t%.1f\n',...
        modelsize1,modelsize2,modelsize3,modelsize4);
    fprintf(fileID,'\tInterval \t[%.3f,%.3f] [%.3f,%.3f] [%.3f,%.3f] [%.3f,%.3f]\n',...
        min(yconf1),max(yconf1),min(yconf2),max(yconf2),min(yconf3),max(yconf3),min(yconf4),max(yconf4));
    fprintf(fileID,'\tCoverage \t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\n',...
        coverage1(i),coverage2(i),coverage3(i),coverage4(i));
    fprintf(fileID,'\tSupport \t%.2f\t\t\t%.2f\t\t\t%.2f\t\t\t%.2f\n',...
    sc1,sc2,sc3,sc4);
    fprintf(fileID,'\tTime \t\t%.3f\t\t\t%.3f\t\t\t%.3f\t\t\t%.3f\n',t1,t2,t3,t4);
end
fprintf(fileID,'%d-fold average coverage is %.3f, %.3f, %.3f, %.3f\n',...
    nruns, mean(coverage1),mean(coverage2),mean(coverage3),mean(coverage4));
fprintf(fileID,'Average inverval length is %.3f, %.3f, %.3f, %.3f\n',...
    mean(conflen1),mean(conflen2),mean(conflen3),mean(conflen4));
fprintf(fileID,'Average number of support computed is %.2f, %.2f, %.2f, %.2f\n',...
    totalsp1/nruns,totalsp2/nruns,totalsp3/nruns,totalsp4/nruns);
fprintf(fileID,'Average time is %.3f, %.3f, %.3f, %.3f\n',time1/nruns,time2/nruns,time3/nruns,time4/nruns);
if ~isequal(fileID,1)
    fclose(fileID);
end






