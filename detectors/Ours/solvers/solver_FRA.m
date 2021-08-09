function [L,S,errHist,tau] = solver_FRA(AY,lambda_S, epsilon, kappa, tenW, opts)
%% Root-finding algorithm for value function
%%
error(nargchk(3,7,nargin,'struct'));
if nargin < 5, opts = []; end
% == PROCESS OPTIONS ==
function out = setOpts( field, default )
    if ~isfield( opts, field )
        opts.(field)    = default;
    end
    out = opts.(field);
    opts    = rmfield( opts, field ); % so we can do a check later
end

A   = @(X) X(:);
signWs = @(x) -1.*(x<=-kappa)+0.*(abs(x)<kappa)+(x>=kappa);
Ws = @(x) 1-x.^2;

[n1,n2] = size(AY);
At  = @(x) reshape(x,n1,n2);


tauInitial      = setOpts('tau0', 1e3 );

L1L2        = setOpts('L1L2',0);
if isempty(L1L2), L1L2=0; end
if L1L2
    if ~isempty(strfind(lower(L1L2),'row')),  L1L2 = 'rows';
    elseif ~isempty(strfind(lower(L1L2),'col')),  L1L2 = 'cols';
        % so col, COL, cols, columns, etc. all acceptable
    else
        error('unrecognized option for L1L2: should be row or column or 0');
    end
end
% Note: setOpts removes things, so add back. Easy to have bugs
opts.L1L2   = L1L2;

finalTol        = setOpts('tol',1e-6);
sumProject      = setOpts('sum', false );
maxProject      = setOpts('max', false );
if (sumProject && maxProject) || (~sumProject && ~maxProject), error('must choose either "sum" or "max" type projection'); end
opts.max        = maxProject; % since setOpts removes them
opts.sum        = sumProject;
rNormOld        = Inf;
tau             = tauInitial;
SPGL1_tol       = setOpts('SPGL1_tol',1e-2);
SPGL1_maxIts    = setOpts('SPGL1_maxIts', 1 );
errHist         = [];
for nSteps = 1:SPGL1_maxIts
%     fprintf(['Iter:' num2str(nSteps) '\n']);
    opts.tol    = max(finalTol, finalTol*10^((4/nSteps)-1) ); % slowly reduce tolerance
    %         opts.tol    = finalTol;
    % fprintf('\n==Running SPGL1 Newton iteration with tau=%.2f\n\n', tau);
    [L,S,errHistTemp] = solver_SPG(AY,lambda_S, tau, kappa, tenW, opts);
    % Find a zero of the function phi(tau) = norm(x_tau) - epsilon
    
    errHist = [errHist; errHistTemp];
    rNorm       = errHist(end,1); % norm(residual)
    if abs( epsilon - rNorm ) < SPGL1_tol*epsilon
        % disp('Reached end of SPGL1 iterations: converged to right residual');
        break;
    end
    if abs( rNormOld - rNorm ) < .1*SPGL1_tol
        % disp('Reached end of SPGL1 iterations: converged');
        break;
    end
    % for nuclear norm matrix completion,
    %   normG = spectralNorm( gradient ) and gradient is just residual
    R   = A(L+S-AY);
    s = signWs(R);
    ws= Ws(s);
    G   = At(ws.*R/(kappa) + s);% it's really two copies, [G;G]
    % if we have a linear operator, above needs to be modified...
    if sumProject
        normG = max(norm(G), (1/lambda_S)*norm(G(:), inf));
    elseif maxProject
        if ~any(L1L2)
            % Using l1 norm for S
            normG = norm(G) +  (1/lambda_S)*norm(G(:), inf);
        elseif strcmpi(L1L2,'rows')
            % dual norm of l1 of l2-norm of rows'
            normG = norm(G) + (1/lambda_S)*norm( sqrt(sum(G.^2,2)), inf );
        elseif strcmpi(L1L2,'cols')
            normG = norm(G) + (1/lambda_S)*norm( sqrt(sum(G.^2,1)), inf );
        end
    end
    
    % otherwise, take another Newton step
    phiPrime    = -normG/rNorm; % derivative of phi
    ALPHA       = .99; % amount of Newton step to take
    tauOld      = tau;
    tau         = tau + ALPHA*( epsilon - rNorm )/phiPrime;
    tau         = min( max(tau,tauOld/10), 1e10 );
%     fprintf(['Tau:' num2str(tau) '\n']);
    % and warm-start the next iteration:
    opts.S0     = S;        
    opts.L0     = L;
end

end % end of main function