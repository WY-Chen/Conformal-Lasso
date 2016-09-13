function [beta,H,lambda] = LTSlassonoglmnet(X,Y,lambdain,alpha,~)
% alpha:    proportion of 'good' data
% mtd:      If empty, run the simple method. else run 50-fold
% initialization

% Parameters
[m,p]=size(X);
h = floor((m+1)*alpha);     % length of chosen set

switch nargin
    case 5
        %% The procedure in paper
        Q = zeros(1,50);
        H = zeros(50,h);
        verboseFlag=1;  % print logs
        % Draw initial guess of H0
        
        % Two C-steps
        wb = waitbar(0,'Initializing C-steps...');
        for i=1:50
            s = randsample(1:m,3);
            beta = lasso(X(s,:),Y(s),'Lambda',lambda,'Standardize',0,'RelTol',1E-4);
            Resid = (Y - X*beta).^2;
            [~,Rind] = sort(Resid);
            H0 = Rind(1:h);
            beta = lasso(X(H0,:),Y(H0),'Lambda',lambda,'Standardize',0,'RelTol',1E-4);
            Resid = (Y - X*beta).^2;
            [Rval,Rind] = sort(Resid);
            H(i,:) = sort(Rind(1:h));
            Q(i) = sum(Rval(1:h))+ h*lambda*sum(abs(beta));
            if verboseFlag==1
               waitbar(i/50,wb,sprintf('Randomizing initial set'));
            end
        end
        close(wb);
        
        % Further C-steps
        [~,s1] = sort(Q);   % Get 10 initializations with smallest error
        s1 = s1(1:10);
        HC = zeros(10,h);
        
        for j=1:10
            i=s1(j);
            Hk = H(i,:);
            stepcount=2;
            while 1
                beta = lasso(X(Hk,:),Y(Hk),'Lambda',lambda,'Standardize',0,'RelTol',1E-12);
                Resid = (Y - X*beta).^2;
                [~,Rind] = sort(Resid);
                Hknew = sort(Rind(1:h));
                if isequal(Hknew,Hk) || stepcount == 20
                    break
                end
                Hk = Hknew;
                stepcount = stepcount+1;
            end
            HC(j,:) = Hk;
%             fprintf('\t%d C-Steps\n',stepcount);
        end
        [uA,~,uIdx] = unique(HC,'rows');
        modeIdx = mode(uIdx);
        H = uA(modeIdx,:)'; %# the first output argument
    case 4
        %% The simple way (without 50-fold)
        Hk = randsample(1:m,h);
        Hk=sort(Hk);
        stepcount = 1;
        while 1
            beta = lasso(X(Hk,:),Y(Hk),...
                'Lambda',lambdain/h,'Standardize',0,'RelTol',1E-8);
            Resid = (Y - X*beta).^2;
            [~,Rind] = sort(Resid);
            Hknew = Rind(1:h);
            Hknew = sort(Hknew);
            if isequal(Hknew,Hk) || stepcount == 20
                break
            end
            Hk = Hknew;
            stepcount = stepcount+1;
        end
%         fprintf('\t%d C-Steps\n',stepcount);
        H=Hk;
end
lambda = lambdain;
        