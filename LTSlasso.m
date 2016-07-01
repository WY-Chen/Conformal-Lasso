function [beta,H,bic] = LTSlasso(X,Y,lambda,alpha)
% alpha:    proportion of 'good' data
[m,p]=size(X);

% Parameters
h = floor((m+1)*alpha);

%% The procedure in paper
% Draw initial guess of H0
% Q = zeros(1,500);
% H = zeros(500,h);
% 
% % Two C-steps
% for i=1:500
%    s = randsample(1:m,3);
%    beta = lasso(X(s,:),Y(s),'Lambda',lambda);
%    Resid = (Y - X*beta).^2;
%    [Rval Rind] = sort(Resid);
%    H0 = Rind(1:h);
%    beta = lasso(X(H0,:),Y(H0),'Lambda',lambda);
%    Resid = (Y - X*beta).^2;
%    [Rval Rind] = sort(Resid);
%    H(i,:) = Rind(1:h);
%    Q(i) = sum(Rval(1:h))+ h*lambda*sum(abs(beta));
% end
% 
% % Further C-steps
% [val s1] = sort(Q);
% s1 = s1(1:10);
% HC = zeros(10,h);
% 
% for i=s1
%     Hk = H(i,:);
%     Hk=sort(Hk);
%     stepcount=2;
%     while 1
%         beta = lasso(X(Hk,:),Y(Hk),'Lambda',lambda);
%         Resid = (Y - X*beta).^2;
%         [Rval Rind] = sort(Resid);
%         Hknew = Rind(1:h);
%         Hknew = sort(Hknew);
%         if isequal(Hknew,Hk) | stepcount == 20
%             break
%         end
%         Hk = Hknew;
%         fprintf('C-Steps %d\n',stepcount);
%         stepcount = stepcount+1;
%     end
%     HC(i,:) = Hk; 
% end
%     
% [nrows t] = size(unique(HC,'rows'));
% H = HC(1,:); %need fix here
%% The simple way (without 500-fold)
Hk = randsample(1:m,h);
Hk=sort(Hk);
stepcount = 1;
while 1
   beta = lasso(X(Hk,:),Y(Hk),'Lambda',lambda);
   Resid = (Y - X*beta).^2;
   [Rval Rind] = sort(Resid);
   Hknew = Rind(1:h);
   Hknew = sort(Hknew);
   if isequal(Hknew,Hk) | stepcount == 20
       break
   end
   Hk = Hknew;
   fprintf('\tC-Steps %d\n',stepcount);
   stepcount = stepcount+1;
end
H=Hk;
mu = sum((Y - X*beta))./h;
sig = sqrt(sum((Y - X*beta - mu).^2)./h);
bic = log(sig) + length(find(beta))*log(m)/m;
    
