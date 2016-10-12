%% Run Testing
% Run testing on a setting with a method. 
%% Implementation
function plotTest(setting)

% Everything vs. model size
if 0
    fprintf(1,'TESTING SETTING %s.\n',setting);
    MS = [1,2,3,4,5,10,15,20,25,30];
    ntests = length(MS);
    c = zeros(ntests,4);
    s = zeros(ntests,4);
    l = zeros(ntests,4);
    for k=1:ntests
        [c(k,:),s(k,:),l(k,:)]=runTest('A','norm',1,MS(k),200,50,.05,.01);
    end
    assignin('base', 'c1', c);assignin('base', 's1', s);assignin('base', 'l1', l);

    f1=figure;
    plot(MS,c);
    ylim([0.8,1]);
    ylabel('Coverage');
    xlabel('True model size');
    legend('Split','Max(Y)-lasso','LOO-lasso','Ridge-lasso','Location','southwest');
    filename = sprintf('Coverage-Modelsize-Setting%s',setting);
    saveas(gcf,fullfile(fullfile(pwd, '\Plots'), filename),'png')

    f2=figure;
    plot(MS,s);
    ylabel('Number of lasso fits');
    xlabel('True model size');
    legend('Split','Max(Y)-lasso','LOO-lasso','Ridge-lasso','Location','northwest');
    filename = sprintf('Support-Modelsize-Setting%s',setting);
    saveas(gcf,fullfile(fullfile(pwd, '\Plots'), filename),'png')


    f3=figure;
    plot(MS,l);
    ylabel('Interval Length');
    xlabel('True model size');
    legend('Split','Max(Y)-lasso','LOO-lasso','Ridge-lasso','Location','northwest');
    filename = sprintf('Length-Modelsize-Setting%s',setting);
    saveas(gcf,fullfile(fullfile(pwd, '\Plots'), filename),'png')

    fclose('all')
end

% Everything vs. sample size
if 0
    fprintf(1,'TESTING SETTING %s.\n',setting);
    SS = [20,50,100,200,400,800];
    ntests = length(SS);
    c = zeros(ntests,4);
    s = zeros(ntests,4);
    l = zeros(ntests,4);
    for k=1:ntests
        [c(k,:),s(k,:),l(k,:)]=runTest('A','norm',1,5,SS(k),50,.05,.01);
    end
    assignin('base', 'c2', c);assignin('base', 's2', s);assignin('base', 'l2', l);

    f1=figure;
    plot(SS,c);
    ylim([0.8,1]);
    ylabel('Coverage');
    xlabel('Sample size');
    legend('Split','Max(Y)-lasso','LOO-lasso','Ridge-lasso','Location','southeast');
    filename = sprintf('Coverage-Samplesize-Setting%s',setting);
    saveas(gcf,fullfile(fullfile(pwd, '\Plots'), filename),'png')

    f2=figure;
    plot(SS,s);
    ylabel('Number of lasso fits');
    xlabel('Sample size');
    legend('Split','Max(Y)-lasso','LOO-lasso','Ridge-lasso','Location','northeast');
    filename = sprintf('Support-Samplesize-Setting%s',setting);
    saveas(gcf,fullfile(fullfile(pwd, '\Plots'), filename),'png')


    f3=figure;
    plot(SS,l);
    ylabel('Interval Length');
    xlabel('Sample size');
    legend('Split','Max(Y)-lasso','LOO-lasso','Ridge-lasso','Location','northeast');
    filename = sprintf('Length-Samplesize-Setting%s',setting);
    saveas(gcf,fullfile(fullfile(pwd, '\Plots'), filename),'png')

    fclose('all')
end

% Everything vs. noise level size
if 1
    fprintf(1,'TESTING SETTING %s.\n',setting);
    sig=[0.1,0.2,0.5,1,2,5];
    ntests = length(sig);
    c = zeros(ntests,4);
    s = zeros(ntests,4);
    l = zeros(ntests,4);
    for k=1:ntests
        [c(k,:),s(k,:),l(k,:)]=runTest('A','norm',sig(k),5,200,50,.05,.01);
    end
    assignin('base', 'c3', c);assignin('base', 's3', s);assignin('base', 'l3', l);

    f1=figure;
    plot(log(sig),c);
    ylim([0.8,1]);
    ylabel('Coverage');
    xlabel('Log-Sigma');
    legend('Split','Max(Y)-lasso','LOO-lasso','Ridge-lasso','Location','southeast');
    filename = sprintf('Coverage-Sigma-Setting%s',setting);
    saveas(gcf,fullfile(fullfile(pwd, '\Plots'), filename),'png')

    f2=figure;
    plot(log(sig),s);
    ylabel('Number of lasso fits');
    xlabel('Log-Sigma');
    legend('Split','Max(Y)-lasso','LOO-lasso','Ridge-lasso','Location','northeast');
    filename = sprintf('Support-Sigma-Setting%s',setting);
    saveas(gcf,fullfile(fullfile(pwd, '\Plots'), filename),'png')


    f3=figure;
    plot(log(sig),l);
    ylabel('Interval Length');
    xlabel('Log-Sigma');
    legend('Split','Max(Y)-lasso','LOO-lasso','Ridge-lasso','Location','northeast');
    filename = sprintf('Length-Sigma-Setting%s',setting);
    saveas(gcf,fullfile(fullfile(pwd, '\Plots'), filename),'png')

    fclose('all')
end
