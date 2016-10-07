function Y = softthresh(v,u)
% soft thresholding
Y = sign(u).*max(abs(u)-v,0);