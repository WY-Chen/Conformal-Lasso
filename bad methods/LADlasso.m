function beta = LADlasso(X,Y,lambda)
[m,p]=size(X);
if nargin < 3
	lambda = pi*sqrt(log(p)/m);
end
X_new = [X;eye(p)*lambda];
Y_new = [Y;zeros(p,1)];
beta = lad(X_new,Y_new);
beta(beta<1E-5)=0;