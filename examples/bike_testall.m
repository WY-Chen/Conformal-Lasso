COVER = zeros(1,108);
addpath(genpath(pwd));

fileID = fopen('bike_ALL_3.txt','w');
for i=101:108
    try
        COVER(i)=bike(i);
    catch ME
        continue
    end
    fprintf(fileID,'%.3f\n',COVER(i));
    fprintf('%d/%d, coverage = %.3f\n', i,108,COVER(i));
end