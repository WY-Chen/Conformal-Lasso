COVER = zeros(1,108);
addpath(genpath(pwd));

fileID = fopen('bike_ALL_3.txt','w');
for i=[1:67,69:108]
    try
        [COVER(i),l]=bike(i);
    catch ME
        continue
    end
    fprintf(fileID,'%.3f\n',COVER(i));
    fprintf('%d/%d, avg. length=%.2f, coverage = %.3f\n', i,108,l,COVER(i));
end

fprintf('Coverage %.2f\n',sum(COVER)/107);