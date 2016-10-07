function [X,Y,xnew,y] = getSetting(setting,tail,sig,nfeatures)
% Setting = 'A', 'B', 'C'.
%       A : Linear. X iid N(0,1). epsilon iid N(0,1). 
%       B : B-spline. X iid N(0,1). epsilon iid t(2).
%       C : Linear. X highly correlated. epsilon iid t(2).

% tail = 'norm', 't'
%    norm : N(0,1)
%       t : t(2)
if strcmp(tail,'norm')
    epsilon1 = normrnd(0,sig,[200,1]);
    epsilon2 = normrnd(0,sig,[10000,1]);
elseif strcmp(tail,'t')
    epsilon1 = trnd(2,[200,1]);
    epsilon2 = trnd(2,[10000,1]);
else
    fprintf('ERROR: No such Tail')
end

beta = [2*(randi([0,1],[1,nfeatures])*2-1),zeros(1,2000-nfeatures)];

if setting=='A'
    X = normrnd(0,1,[200,2000]);
    Y = X*beta'+epsilon1;
    xnew = normrnd(0,1,[1,2000]);
    y = xnew*beta'+epsilon2;
elseif setting=='D'
     X = normrnd(0,1,[200,2000]);
     Y = 2*X(:,1)+2*X(:,2).^2+2*X(:,3).^4+epsilon1;
     meanY = mean(Y); Y=Y-meanY;
     xnew = normrnd(0,1,[1,2000]);
     y = 2*xnew(:,1)+2*xnew(:,2).^2+2*xnew(:,3).^4+epsilon2;
     y=(y-meanY)./1;
elseif setting=='E'
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
elseif setting=='B'
    S = repmat(1:2000,[2000,1]);S = bsxfun(@minus,S,(1:2000)');
    S = power(.9,abs(S));
    D = mvnrnd(zeros(201,2000),S);
    X = D(1:200,:);
    Y = X*beta'+epsilon1;
    xnew = D(201,:);
    y = xnew*beta'+epsilon2;
elseif setting=='C'
    a = rand(2000,2000); a = a- mean(a(:)); ata = a'*a;
    sigma = ata/max(abs(ata(:)));
    D = mvnrnd(zeros(201,2000),sigma);
    X = D(1:200,:);
    Y = X*beta'+epsilon1;
    xnew = D(201,:);
    y = xnew*beta'+epsilon2;
elseif setting=='F'
    X1 = normrnd(0,1,[100,2000]);
    X2= trnd(2,[100,2000]);
    X=[X1; X2];
    Y = X*beta'+epsilon1;
    xnew = normrnd(0,1,[1,2000]);
    y = xnew*beta'+epsilon2;
elseif setting=='G'
    X = normrnd(0,1,[200,2000]);

    perms=randperm(200);
    choose=perms(1:5); %choose which observations to be perturbed

    for i=1:5
        add=randi(5,1,2000); 
        X(choose(i),:)=X(choose(i),:)+add; %add an integer of 1-5 for each selected observation
    end

    Y = X*beta'+epsilon1;
    xnew = normrnd(0,1,[1,2000]);
    y = xnew*beta'+epsilon2;
else
    fprintf('ERROR: No such Setting')
end 