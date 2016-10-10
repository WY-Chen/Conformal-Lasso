function [coverage,support,length]=runTest(setting,tail,sig,nfeatures,n,nruns,alpha,stepsize)
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
    fprintf(2,'TESTING=== run %d/%d.\n',i,nruns);
    
    % Get testing data
    [X,Y,xnew,y] = getSetting(setting,tail,sig,nfeatures,n);
    X_withnew = [X;xnew];

    % Get lambda from empirical expectation
    if strcmp(tail,'norm')
        lambda = mean(arrayfun(@(t) norm(X_withnew'*normrnd(0,1,[n+1,1]),inf)*2, 1:1000));
    else
        lambda = mean(arrayfun(@(t) norm(X_withnew'*trnd(2,[n+1,1]),inf)*2, 1:1000));
    end
    rangeY = max(Y)-min(Y);
    ytrial = (min(Y)-2*rangeY):stepsize:(max(Y)+2*rangeY);

    % run method
    tic;
    [yconf1,~,sc1] = conformalLassoSplit(X,Y,xnew,alpha,[],lambda);
    t1=toc;time1=time1+t1;tic;
    [yconf2,~,sc2] = conformalLassoCtnFit(X,Y,xnew,alpha,min(Y):stepsize:max(Y),lambda);
    t2=toc;time2=time2+t2;tic;
    [yconf3,~,sc3,~] = conformalLassoTruncate_LOO(X,Y,xnew,alpha,ytrial,lambda);
    t3=toc;time3=time3+t3;tic;
    [yconf4,~,sc4,~] = conformalLassoTruncate_ridge(X,Y,xnew,alpha,ytrial,lambda);
    t4=toc;time4=time4+t4;
    totalsp1 = totalsp1+sc1;
    totalsp2 = totalsp2+sc2;
    totalsp3 = totalsp3+sc3;
    totalsp4 = totalsp4+sc4;
    if isempty(yconf1)
        yconf1=ytrial;
        fprintf(2,'Empty yconf\n');
    end
    if isempty(yconf2)
        yconf2=ytrial;
        fprintf(2,'Empty yconf\n');
    end
    if isempty(yconf3)
        yconf3=ytrial;
        fprintf(2,'Empty yconf\n');
    end
    if isempty(yconf4)
        yconf4=ytrial;
        fprintf(2,'Empty yconf\n');
    end
    coverage1(i) = sum((min(yconf1)<y)&(y<max(yconf1)))/10000;
    coverage2(i) = sum((min(yconf2)<y)&(y<max(yconf2)))/10000;
    coverage3(i) = sum((min(yconf3)<y)&(y<max(yconf3)))/10000;
    coverage4(i) = sum((min(yconf4)<y)&(y<max(yconf4)))/10000;
    conflen1(i) = max(yconf1)-min(yconf1);
    conflen2(i) = max(yconf2)-min(yconf2);
    conflen3(i) = max(yconf3)-min(yconf3);
    conflen4(i) = max(yconf4)-min(yconf4);
end
fprintf(1,'%d-fold average coverage is %.3f, %.3f, %.3f, %.3f\n',...
    nruns, mean(coverage1),mean(coverage2),mean(coverage3),mean(coverage4));
fprintf(1,'Average inverval length is %.3f, %.3f, %.3f, %.3f\n',...
    mean(conflen1),mean(conflen2),mean(conflen3),mean(conflen4));
fprintf(1,'Average number of support computed is %.2f, %.2f, %.2f, %.2f\n',...
    totalsp1/nruns,totalsp2/nruns,totalsp3/nruns,totalsp4/nruns);
fprintf(1,'Average time is %.3f, %.3f, %.3f, %.3f\n',time1/nruns,time2/nruns,time3/nruns,time4/nruns);

coverage = [mean(coverage1),mean(coverage2),mean(coverage3),mean(coverage4)];
support  = [totalsp1/nruns,totalsp2/nruns,totalsp3/nruns,totalsp4/nruns];
length   = [mean(conflen1),mean(conflen2),mean(conflen3),mean(conflen4)];