function [X,Y,xnew,y] = getSetting(setting)
% Setting = 'A', 'B', 'C'.
%       A : Linear. X iid N(0,1). epsilon iid N(0,1). 
%       B : B-spline. X iid N(0,1). epsilon iid t(2).
%       C : Linear. X highly correlated. epsilon iid t(2).
if setting=='A'
    beta = [2,2,2,2,2,zeros(1,1995)];
    X = normrnd(0,1,[200,2000]);
    Y = X*beta'+normrnd(0,1,[200,1]);
    xnew = normrnd(0,1,[1,2000]);
    y = xnew*beta'+normrnd(0,1,[10000,1]);
elseif strcmp(setting,'Astrong')
    beta = [2,2,2,2,2,zeros(1,1995)]*20;
    X = normrnd(0,1,[200,2000]);
    Y = X*beta'+normrnd(0,1,[200,1]);
    xnew = normrnd(0,1,[1,2000]);
    y = xnew*beta'+normrnd(0,1,[10000,1]);
elseif strcmp(setting,'Asmall')
    beta = [2,2,2,2,2,zeros(1,1995)]*20;
    X = normrnd(0,1,[50,2000]);
    Y = X*beta'+normrnd(0,1,[50,1]);
    xnew = normrnd(0,1,[1,2000]);
    y = xnew*beta'+normrnd(0,1,[10000,1]);
elseif setting=='B'
     X = normrnd(0,1,[200,2000]);
     Y = 3*X(:,1).^2+2*X(:,2).^3+5*X(:,3).^4+trnd(2,[200,1]);
     xnew = normrnd(0,1,[1,2000]);
     y = 3*xnew(:,1).^2+2*xnew(:,2).^3+5*xnew(:,3).^4+trnd(2,[10000,1]);
elseif setting=='C'
    beta = [2,2,2,2,2,zeros(1,1995)];
    X = randi([1,3],201,2000);
    X(X==1)=normrnd(0,1,[1,length(X(X==1))]);
    X(X==2)=pearsrnd(0,1,1,5,[1,length(X(X==2))]);
    X(X==3)=binornd(1,.5,[1,length(X(X==3))]);
    for k=1:2000
       X(:,k)=0.4*X(:,k)+0.3*X(:,max(1,k-1)) + 0.2*X(:,max(1,k-2)) + 0.1*X(:,max(1,k-3));
    end
    xnew = X(201,:);
    X = X(1:200,:);
    Y = X*beta' + trnd(2,[200,1]);
    y = xnew * beta'+trnd(2,[10000,1]);
else
    fprintf('ERROR: No such Setting')
end 