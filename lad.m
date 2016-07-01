function [beta, se, cov] = lad(x, y, c, p)% LAD(X, Y, c, p) runs a linear least absolute deviations regression of % the vector Y on the matrix X and reports the results. % If c is 1, a vector of ones is appended to X.% If p = 0, reporting of results is suppressed. %% LAD uses the asymptotic standard errors of Bassett and Koenker, 1978,% "Asymptotic Theory of Least Absolute Error Regresssion", JASA, 618--622.%% LAD uses the linear programming algorithm of Morillo and Koenker, % Ludger Hentschel % Mon, Feb 18, 2002%--- set defaults ---%if nargin < 4	p = 1;	if nargin < 3		c = 0;	endend%--- handle inclusion/exclusion of constant ---%[t, k] = size(x);         % t = # of obs., k = # of regressorsif c == 1  x = [ones(t, 1) x];  k = k + 1;else  if all(x(:,1) == 1)    c = 1;  else    c = 0;  endend%--- check for input errors ---%[m, n] = size(y);if n > 1  error('Y must be a vector.');endif m~=t  error('Y and X must have the same number of rows.');endif k > t  error('Fewer observations than degrees of freedom.');elseif k == t  warning('Number of observations equals number of parameters.');end%--- run the regression and compute statistics ---%% Estimate the coefficients using linear programming.% This seems stable and fast.b = rq_fnm(x, y, .5);% An alternative is to estimate the regression through iteration.% The goal is to find b = inv(x'*z*x)*x'*z*y, where z = inv(diag( abs(e(b)) )).% Do this by finding b(i) based on z(i-1) until b(i) = b(i-1) +/- tol.% This is fast in many situations. It tends to converge very slowly when the % columns of x are highly correlated, but long before OLS would complain about % multicollinearity.%% tol = 1e-4;% maxiter = 500*k;% b = x\y;  % start with ols estimates% % % If x is poorly conditioned, it takes forever to find the LAD% % estimates. Yet, the predictions are not sensitive to the coefficients,% % so we can simply use the OLS coefficients.% i = 0;% if cond(x) > 5000% else% 	b0 = b + 1;% 	while (i < maxiter) & any(abs(b - b0) > tol)% 		b0 = b;% 		e = y - x*b;% 	%	z = repmat(1./abs(e), 1, k); % seems to be slower than "* ones(1, k)"% 		z = 1./abs(e) * ones(1, k);% 		b = inv((x .* z)' * x) * ((x .* z)' * y);% 		i = i + 1;% 	end% end% % if i >=  maxiter% 	warning('LAD estimates did not converge.');% end%--- Find standard errors for coefficients ---%% Coefficient estimates are asymptotically normal.% Asymptotic standard errors depend on the distribution of the errors.% I assume that errors are normal, but this setting largely defeats the  % point of LAD regression. A better version of this function would % let the user specify the distribution of the errors, or estimate f(0)% nonparametrically.% Different distributions make non-trivial differences to the standard errors.beta = b;[qx, r] = qr(x, 0);xxi  = r \ (r' \ eye(k));           % numerically stable inverse of x'x% w  = (2*pdfn(0)).^(-1) = sqrt(2*pi)/2 depends on F(e), assume normalityw2    = (2*pi)/4;cov = w2 * xxi; se   = sqrt(diag(cov));    % standard errors%--- The following is code for quantile regressions ---%% The functions implement the Morillo & Koenker algorithm for % quantile regressions. The algorithm uses linear programming techniques.%% A .5 quantile regression is equivalent to least absolute deviations.%function b = rq_fnm(X, y, p)% Construct the dual problem of quantile regression% Solve it with lp_fnm% % Function rq_fnm of Daniel Morillo & Roger Koenker% Found at: http://www.econ.uiuc.edu/~roger/rqn/rq.ox% Translated from Ox to Matlab by Paul Eilers 1999% Modified slightly by Roger Koenker 2000-1%[m n] = size(X);u = ones(m, 1);a = (1 - p) .* u;b = -lp_fnm(X', -y', X' * a, u, a)';function y = lp_fnm(A, c, b, u, x)% Solve a linear program by the interior point method:% min(c * u), s.t. A * x = b and 0 < x < u% An initial feasible solution has to be provided as x%% Function lp_fnm of Daniel Morillo & Roger Koenker% Found at: http://www.econ.uiuc.edu/~roger/rqn/rq.ox% Translated from Ox to Matlab by Paul Eilers 1999% Set some constantsbeta   = 0.9995;small  = 1e-5;max_it = 50;% beta   = 0.999995;% small  = 1e-7;% max_it = 150;[m n] = size(A);% Generate initial feasible points = u - x;y = (A' \  c')';r = c - y * A;z = r .* (r > 0);w = z - r;gap = c * x - y * b + w * u;% Start iterationsit = 0;while gap > small & it < max_it	it = it + 1;	% Compute affine step	q   = 1 ./ (z' ./ x + w' ./ s);	r   = z - w;	Q   = sparse(1:n,1:n,q);	AQ  = A * Q;	AQA = AQ * A';	rhs = AQ * r';	dy  = (AQA \ rhs)';	dx  = q .* (dy * A - r)';	ds  = -dx;	dz  = -z .* (1 + dx ./ x)';	dw  = -w .* (1 + ds ./ s)';				% Compute maximum allowable step lengths	fx = bound(x, dx);	fs = bound(s, ds);	fw = bound(w, dw);	fz = bound(z, dz);	fp = min(fx, fs);	fd = min(fw, fz);	fp = min(min(beta * fp), 1);	fd = min(min(beta * fd), 1);		% If full step is feasible, take it. Otherwise modify it	if min(fp, fd) < 1    		% Update mu		mu = z * x + w * s;		g = (z + fd * dz) * (x + fp * dx) + (w + fd * dw) * (s + fp * ds);		mu = mu * (g / mu) ^3 / ( 2* n);		% Compute modified step		dxdz = dx .* dz';		dsdw = ds .* dw';		xinv = 1 ./ x;		sinv = 1 ./ s;		xi   = mu * (xinv - sinv);		rhs  = rhs + A * ( q .* (dxdz - dsdw -xi));		dy   = (AQA \ rhs)';		dx = q .* (A' * dy' + xi - r' -dxdz + dsdw);		ds = -dx;		dz = mu * xinv' - z - xinv' .* z .* dx' - dxdz';		dw = mu * sinv' - w - sinv' .* w .* ds' - dsdw';		% Compute maximum allowable step lengths		fx = bound(x, dx);		fs = bound(s, ds);		fw = bound(w, dw);		fz = bound(z, dz);		fp = min(fx, fs);		fd = min(fw, fz);		fp = min(min(beta * fp), 1);		fd = min(min(beta * fd), 1);	end	% Take the step	x = x + fp * dx;	s = s + fp * ds;	y = y + fd * dy;	w = w + fd * dw;	z = z + fd * dz;	gap = c * x - y * b + w * u;	%disp(gap);endif it >= max_it	warning('LAD did not converge.');endfunction b = bound(x, dx)% Fill vector with allowed step lengths% Support function for lp_fnmb = 1e20 + 0 * x;f = find(dx < 0);b(f) = -x(f) ./ dx(f);