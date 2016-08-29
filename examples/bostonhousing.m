%% Import data from text file.
% Script for importing data from the following text file:
%
%    E:\Github\Conformal-Lasso\examples\housing.data
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2016/08/21 18:21:56

%% Initialize variables.
filename = 'housing.data';

%% Format string for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%8f%7f%8f%3f%8f%8f%7f%8f%4f%7f%7f%7f%7f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '',  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
housing = [dataArray{1:end-1}];
%% Clear temporary variables
clearvars filename formatSpec fileID dataArray ans;

fileID = 1;

Xtot = housing(:,1:13);
Ytot = housing(:,end);

Xtot = Xtot./max(Xtot(:));
Xtot(:,4)=housing(:,4);
meanYtot = mean(Ytot);
Ytot = Ytot-meanYtot;
maxYtot = max(Ytot);
Ytot = Ytot./maxYtot;

% m=56;M=506;
% train = randsample(1:M,m);
% Xtrain = Xtot(train,:);
% Ytrain = Ytot(train);
% 
% test = setxor(1:M,train);
% Xtest = Xtot(test,:);
% Ytest = Ytot(test);

incounter1 = 0;incounter2 = 0;
L1 = []; U1 = [];
L2 = []; U2 = [];

test = randsample(1:506,200);
for i=1:200
    xnew = Xtot(test(i),:);
    train = setxor(1:506,test(i));
    Xtrain = Xtot(train,:);
    Ytrain = Ytot(train);    
    
    X_withnew = [Xtrain;xnew];
    y = Ytot(test(i));
    ytrial = -1:0.01:1;  
    
    [yconf1,ms1,~] = conformalLOO(Xtrain,Ytrain,xnew,.1,ytrial,0.0125);
    [yconf2,~,~] = conformal(Xtrain,Ytrain,xnew,.1,'linear',ytrial);
    fprintf(fileID,'Prediction interval: LOO: [%.2f,%.2f] OLS:[%.2f, %.2f]\n',...
        min(yconf1)*maxYtot+meanYtot,max(yconf1)*maxYtot+meanYtot,...
        min(yconf2)*maxYtot+meanYtot,max(yconf2)*maxYtot+meanYtot);
    if (min(yconf1)<=y)&&(y<=max(yconf1))
        incounter1=incounter1+1;
        fprintf(fileID,'Real data is LOO: IN  ');
    else
        fprintf(fileID,'Real data is LOO: OUT  ');
    end
    if (min(yconf2)<=y)&&(y<=max(yconf2))
        incounter2=incounter2+1;
        fprintf(fileID,'OLS: IN\n');
    else
        fprintf(fileID,'OLS: OUT\n');
    end
    L1 = [L1 min(yconf1)*maxYtot+meanYtot];
    U1 = [U1 max(yconf1)*maxYtot+meanYtot];
    L2 = [L2 min(yconf2)*maxYtot+meanYtot];
    U2 = [U2 max(yconf2)*maxYtot+meanYtot];
    disp(i/200);
end
plot(1:200,Ytot(test)*maxYtot+meanYtot,'bo','MarkerFaceColor','b');
set(gca, 'color', [.85 .85 .85]);
hold on;
plot([find(U1'-Ytot(test)*maxYtot-meanYtot<0)'...
    find(L1'-Ytot(test)*maxYtot-meanYtot>0)'],...
    Ytot(test(([find(U1'-Ytot(test)*maxYtot-meanYtot<0)'...
    find(L1'-Ytot(test)*maxYtot-meanYtot>0)'])))*maxYtot+meanYtot,'ro','MarkerFaceColor','r');
plot([find(U2'-Ytot(test)*maxYtot-meanYtot<0)'...
    find(L2'-Ytot(test)*maxYtot-meanYtot>0)'],...
    Ytot(test(([find(U2'-Ytot(test)*maxYtot-meanYtot<0)'...
    find(L2'-Ytot(test)*maxYtot-meanYtot>0)'])))*maxYtot+meanYtot,'ro','MarkerFaceColor','r');
for i=1:200
    pl1=line([i i], [L1(i) U1(i)],'LineWidth',3);
    pl2=line([i i], [L2(i) U2(i)],'LineWidth',3);
    pl3=line([i i], [max(L2(i),L1(i)) min(U1(i),U2(i))],'LineWidth',3);
    pl1.Color=([1,1,0,1]);pl2.Color =([0,1,1,1]);pl2.Color =([0,1,0,1]);
end
title('Conformal Prediction intervals');
legend([pl1,pl2,pl3],'LOO','OLS','Overlapping');
hold off;
fprintf(fileID,'The coverage is LOO:%.3f OLS:%.3f\n',incounter1/200,incounter2/200);
fprintf(fileID,'Average interval length is LOO:%.3f OLS:%.3f\n',mean(U1-L1),mean(U2-L2));
fprintf(fileID,'Median interval length is LOO:%.3f OLS:%.3f\n',median(U1-L1),median(U2-L2));
fprintf(fileID,'LOO model size %.2f\n',ms1);

