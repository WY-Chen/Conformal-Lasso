function [X,Y,xnew,y] = getSetting(setting,tail)
% Setting = 'A', 'B', 'C'.
%       A : Linear. X iid N(0,1). epsilon iid N(0,1). 
%       B : B-spline. X iid N(0,1). epsilon iid t(2).
%       C : Linear. X highly correlated. epsilon iid t(2).
if strcmp(tail,'norm')
    epsilon1 = normrnd(0,1,[200,1]);
    epsilon2 = normrnd(0,1,[10000,1]);
elseif strcmp(tail,'t')
    epsilon1 = trnd(2,[200,1]);
    epsilon2 = trnd(2,[10000,1]);
else
    fprintf('ERROR: No such Tail')
end

if setting=='A'
    beta = [-2,-2,-2,-2,-2,zeros(1,1995)];
    X = normrnd(0,1,[200,2000]);
    Y = X*beta'+epsilon1;
    xnew = normrnd(0,1,[1,2000]);
    y = xnew*beta'+epsilon2;
elseif setting=='B'
     X = normrnd(0,1,[200,2000]);
     Y = 2*X(:,1)+2*X(:,2).^2+2*X(:,3).^4+epsilon1;
     meanY = mean(Y); Y=Y-meanY;
     xnew = normrnd(0,1,[1,2000]);
     y = 2*xnew(:,1)+2*xnew(:,2).^2+2*xnew(:,3).^4+epsilon2;
     y=(y-meanY)./1;
elseif setting=='C'
    beta = [2,2,2,2,2,zeros(1,1995)];
    X = randi([1,3],201,2000);
    X(X==1)=normrnd(0,1,[1,length(X(X==1))]);
    X(X==2)=pearsrnd(0,1,1,5,[1,length(X(X==2))]);
    X(X==3)=binornd(1,.5,[1,length(X(X==3))]);
    for k=1:2000
       X(:,k)=0.4*X(:,k)+0.3*X(:,max(1,k-1)) + 0.2*X(:,max(1,k-2)) + 0.1*X(:,max(1,k-3));
    end
    X = X-mean(X(:));
    xnew = X(201,:);
    X = X(1:200,:);
    Y = X*beta' + epsilon1;
    mY=mean(Y); 
    Y= Y-mY;
    y = xnew * beta'+epsilon2-mY;
elseif setting=='D'
    beta = [2,2,2,2,2,zeros(1,1995)];
    D = mvnrnd(zeros(201,2000),ones(2000,2000)*0.6+0.4*diag(ones(1,2000)));
    X = D(1:200,:);
    Y = X*beta'+epsilon1;
    xnew = D(201,:);
    y = xnew*beta'+epsilon2;
elseif setting=='E'
    beta = [2,2,2,2,2,zeros(1,1995)];
    a = rand(2000,2000); a = a- mean(a(:)); ata = a'*a;
    sigma = ata/max(abs(ata(:)));
    D = mvnrnd(zeros(201,2000),sigma);
    X = D(1:200,:);
    Y = X*beta'+epsilon1;
    xnew = D(201,:);
    y = xnew*beta'+epsilon2;
else
    fprintf('ERROR: No such Setting')
end 