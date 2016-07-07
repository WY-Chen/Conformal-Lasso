function [mini,maxi]=solveInt(A,b,Y)
% solve for a single variable system of inequalities
[p,m]=size(A);
para = A(:,m);
bound = b-A(:,1:(m-1))*Y;

pos = find(para>0);
neg = setxor(pos,1:p);
if ~isempty(pos)
    maxi = min(bound(pos)./para(pos));
else
    maxi = inf;
end

if ~isempty(neg)
    mini = max(bound(neg)./para(neg));
else 
    mini = -inf;
end