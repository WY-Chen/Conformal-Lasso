%% Run Testing
% Run testing on a setting with a method. 
%% Implementation
function plotTest(setting)

fileID = 2;
nruns  = 100;
alpha  = 0.5;

% Everything vs. model size
fprintf(fileID,'TESTING SETTING %s.\n',setting);
MS = [1,5,10,20,30];
ntests = length(MS);
c = zeros(ntests,4);
s = zeros(ntests,4);
l = zeros(ntests,4);
for k=1:ntests
    [c(k,:),s(k,:),l(k,:)]=runTest('A','norm',1,MS(k),200,10,.05,.01);
end
assignin('base', 'c', c);assignin('base', 's', s);assignin('base', 'l', l);

plot(MS,c);
ylim([0.8,1]);
ylabel('Coverage');
xlabel('True model size');
legend('Split','Yrange-lasso','LOO-lasso','Ridge-lasso','Location','southwest');


