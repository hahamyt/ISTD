function res = run_Ours(img)
    %% Config
%     addpath('./solvers');
%     addpath('./utils');
    % priori weight 
    point_target = true;
    patchSize = 25; 
    slideStep = 25;
    % max version objective function settings
    lambda      = 2e-2;%2e0;                        % 2e-2;
    kappa       = 30;                               % Huber peanlty
    sum_ = false;
    max_ = true;
    tau0 = 2e5;     % 2e5 the bigger, the more noise points
    SPGL1_tol = 1e-5;
    tol_ = 1e-5;
    printEvery = 0;
    
    gpu = true;
    
    %% Preprocessing
    img = preprocess(img);
    
    %% constrcut patch tensor of original image
    if gpu
        tenD = gpuArray(gen_patch_ten(img, patchSize, slideStep));
    else
        tenD = gen_patch_ten(img, patchSize, slideStep);
    end
    [n1,n2,n3] = size(tenD);
    X = reshape(tenD, n1*n2, n3);
    nPatches = size(X,2);
    L0 = repmat(median(X,2), 1, nPatches);  % X - S0;           
    S0 = X - L0;                            % zeros(m, n);
    epsilon = 5e-3*norm(X,'fro');           % tolerance for fidelity to data
    opts = struct('sum',sum_,'L0',L0,'S0',S0,'max',max_,'tau0',tau0,...
                'SPGL1_tol',SPGL1_tol,'tol',tol_,'printEvery',printEvery, 'gpu', gpu);
    %% calculate prior weight map
    if point_target
        %      step 1: calculate two eigenvalues from structure tensor
        [lambda1, lambda2] = structure_tensor_lambda(img, 3);
        %      step 2: calculate corner strength function
        cornerStrength = (((lambda1.*lambda2)./(lambda1 + lambda2)));
        %      step 3: obtain final weight map
        maxValue = (max(lambda1,lambda2));
        priorWeight = mat2gray(cornerStrength .* maxValue);
        %      step 4: constrcut patch tensor of weight map
        if gpu
            tenW = gpuArray(gen_patch_ten(priorWeight, patchSize, slideStep));
        else
            tenW = gen_patch_ten(priorWeight, patchSize, slideStep);
        end
    else
        if gpu
            tenW = gpuArray(ones(n1, n2, n3));
        else
            tenW = ones(n1, n2, n3);    
        end
    end
    %% Run the SPGL1 algorithm
    %         tic;
    [~, S] = solver_FRA(X,lambda,epsilon,kappa, tenW, opts);
    %         tt = toc;
    %% recover from patch tensor
    res = res_patch_ten_mean(reshape(S, n1, n2,n3), img, patchSize, slideStep);
%     B = res_patch_ten_mean(reshape(B, n1, n2,n3), img, patchSize, slideStep);
%     B = gather(B);imshow(B/255);    
%     se=strel('square', 7);
%     res=imdilate(gather(res), se);  % +rand(size(res));
end
    
