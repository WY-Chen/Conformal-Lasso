function Y = softthresh(v,u)

Y = sign(u).*max(abs(u)-v,0);