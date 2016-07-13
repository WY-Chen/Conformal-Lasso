% import data, use GDP of 1980-2014 to predict 2015
data=csvread('worldGDP.csv');
[m,p]=size(data);

% format data
for i=1:m
    data(i,:)=(data(i,:)-data(i,1))./(max(data(i,1:(p-1)))-min(data(i,1:(p-1))));
end

% use 30 countries as training data
sample = randsample(1:m,30);
X = data(sample,1:p-1);
Y = data(sample,p);
test = setxor(1:m,sample);

incounter = 0;

for i=test
    xnew = data(i,1:p-1);
    y = data(i,p);
    ytrial = [-2:0.001:2];
    lambda=0.1;
    [yconf,modelsize] = conformalLTSLassoAllSupp(X,Y,xnew,0.1,ytrial,lambda);
    fprintf('Prediction interval is [%.3f,%.3f] with model size %d while real data is %.3f\n',...
        min(yconf),max(yconf),modelsize,y);
    if (min(yconf)<y)&&(y<max(yconf))
        incounter=incounter+1;
        fprintf('Real data is IN\n');
    else
        fprintf('Real data is OUT\n');
    end
end

fprintf('The coverage is %.2f\n',incounter/(m-30));