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
elseif setting=='B'
     X = normrnd(0,1,[200,2000]);
     Y = 3*X(:,1).^2+2*X(:,2).^3+X(:,3).^4+trnd(2,[200,1]);
     xnew = normrnd(0,1,[1,2000]);
     y = 3*xnew(:,1).^2+2*xnew(:,2).^3+xnew(:,3).^4+trnd(2,[10000,1]);
elseif setting=='C'
    X = zeros(200,2000);
    X(1:66,:)=normrnd(0,1,66,2000);
    X(67:132,:)=pearsrnd(0,1,1,5,[66,2000]);
    X(133:200,:)=binornd(1,0.5,[68,2000]);
    X=X(:, randperm(size(X,2)));
    X=X(randperm(size(X,1)), :);
    for k=1:2000
       X(:,k)=0.3*X(:,max(1,k-1)) + 0.5*X(:,max(1,k-2)) + 0.2*X(:,max(1,k-3));
    end
    Y = X*beta' + trnd(2,[200,1]);
    xnew = normrnd(0,1,[1,2000]);
    y = xnew * beta'+trnd(2,[10000,1]);
else
    fprintf('ERROR: No such Setting')
end 