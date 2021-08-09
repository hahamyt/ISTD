%% Config
addpath('./experiments');
addpath('./utilities');
clear all;clc;
root_dir = pwd;
%% Visulation or not
viz = false;
%% Necessary settings
detectors = { 'TopHat', 'ADDGD', 'ADMD', 'HBMLCM', 'LEF', 'LIG', 'WSLCM', 'PSTNN', 'IPI', 'Ours'}; %

seqs = {'seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6'};
thresholds = 1:0.01:1;

[~, num_seq] = size(seqs);
[~, num_det] = size(detectors);
%% SCRG, CG, and BSF computing
% SCRG = (S/C)_out/(S/C)_in
% BSF = C_in/C_out
% CG = S_out/ S_in
SCRG = zeros(num_seq, num_det);
BSF = zeros(num_seq, num_det);
CG = zeros(num_seq, num_det);

%% Start detection sequences
fprintf('Detecting\n');
for dd = 1:num_det
    fprintf('Detector: %s\n', detectors{dd});
    for s = 1:num_seq
        flag = 0;
        %% SCRG
        if exist(['result/SCRG/' seqs{s} '_' detectors{dd} '_SCRG.mat'], 'file')
            load(['result/SCRG/' seqs{s} '_' detectors{dd} '_SCRG.mat']);
            SCRG(s, dd) = scrg_tmp;
            flag = flag + 1;
        end
        %% BSF
        if exist(['result/BSF/' seqs{s} '_' detectors{dd} '_BSF.mat'], 'file')
            load(['result/BSF/' seqs{s} '_' detectors{dd} '_BSF.mat']);
            BSF(s, dd) = bsf_tmp;
            flag = flag + 1;
        end
        %% CG
        if exist(['result/CG/' seqs{s} '_' detectors{dd} '_CG.mat'], 'file')
            load(['result/CG/' seqs{s} '_' detectors{dd} '_CG.mat']);
            CG(s, dd) = cg_tmp;
            flag = flag + 1;
        end
        if flag == 3
            continue;
        end 
        %%
        fprintf('Seq: %s\n', seqs{s});
        gts = load(['data/groundturth/' num2str(seqs{s}) '.txt']);
        if ~exist(['result/' seqs{s} '/' detectors{dd}], 'dir')
           run_experiment; 
        end
        d = dir(['./data/' seqs{s}]);
        nameCell = cell(length(d)-2,1);  % arrange the order of the seq
        for i = 3:length(d)
        %     disp(d(i).name)
            nameCell{i-2} = strcat(d(i).folder,'/', d(i).name);%d(i).name;
        end
        imglist = sort_nat(nameCell);
        %% Start detection 
        img=imread(imglist{1});     % all images must have the same size
        if size(img, 3) == 3
            img = double(rgb2gray(img));
        end
        img = double(img);
        [m, n] = size(img); 
        %% For ROC
        responses = cell(length(imglist), 1);   % zeros(length(imglist), m, n);
        gtms = zeros(length(imglist), m, n);
       %% For SCRG
        S_in = zeros(length(imglist), 1); % Every frame in every seq has its amplitude
        C_in = zeros(length(imglist), 1); % Every frame has its standard deviation of the clutter
        
        S_out = zeros(length(imglist), 1); % Every response of every frame has its amplitude
        C_out = zeros(length(imglist), 1); % Every response has its standard deviation of the clutter
        
        %% load the pregenerated raw responses: for faster evaluation
        raw_res = ['./result/' seqs{s} '/' detectors{dd} '/raw_responses'];
        if ~exist(raw_res)
            mkdir(raw_res);
        end
        have_raw = 0;
        if exist([raw_res '/responses.mat'])
           load([raw_res '/responses.mat']);
           have_raw = 1;
        end
        
        for k=1:length(imglist)
            img=imread(imglist{k});
%             [imw, imh] = size(img);
            if size(img, 3) == 3
                img = double(rgb2gray(img));
            end
            img = double(img);
            
           %% For SCRG
            img_ = img;
%             img_ = mapminmax(img_, 0, 1);
            gt = gts(k, :);
            % ground truth region
            error("You should set the ground-truth region parameter {w} and {scale} manually.");
            w = 0;scale = 0;
            region_target = img_(gt(1)-w:gt(1)+w, gt(2)-w:gt(2)+w);
            img_(gt(1)-w:gt(1)+w, gt(2)-w:gt(2)+w) = 0;
            region_around = img_; %(max(gt(1)-scale*w,1):min(gt(1)+scale*w,imw),...
                                  %max(gt(2)-scale*w,1):min(gt(2)+scale*w, imh));
            
            tmp = mean(region_around);tmp(isnan(tmp))=0;
            S_in(k) = abs(mean(mean(region_target))- mean(tmp));% max(max(region_in)); % img_(gt(1)-w:gt(1)+w, gt(2)-w:gt(2)+w)));                % amplitude
            C_in(k) = std(region_around(:));                      % standard deviation

           %% Run: Collecting responses
           if ~have_raw
                % preds
                tic;
                funcName = ['response=run_' detectors{dd} '(img);'];
                cd(['./detectors/' detectors{dd}]);
                addpath(genpath('./'));
                eval(funcName);
                tt = toc;
                cd(root_dir);
                response = gather(response);
                responses{k} = response;
                fprintf('%d/%d, time: %3s \n', k, length(imglist), tt);
           else
                response = responses{k}; 
%                 response=(response-min(response(:)))/(max(response(:))-min(response(:)));
           end
       
           %% For SCRG
            region_target = response(gt(1)-w:gt(1)+w, gt(2)-w:gt(2)+w);
            response(gt(1)-w:gt(1)+w, gt(2)-w:gt(2)+w) = 0;
            region_around = response;   %(max(gt(1)-scale*w,1):min(gt(1)+scale*w,imw),...
                                        %max(gt(2)-scale*w,1):min(gt(2)+scale*w, imh));
            
            tmp = mean(region_around);tmp(isnan(tmp))=0;
            S_out(k) = abs(mean(mean(region_target))- mean(tmp)); % max(max(region_out)); %mean(response(gt(1)-2:gt(1)+2, gt(2)-2:gt(2)+2));                % amplitude
            C_out(k) = std(region_around(:));                  % standard deviation
            
            clear response;
            
            if viz
                figure(1);
                subplot(121);
                imshow(img./255);
                subplot(122);
                imagesc(squeeze(responses(k, :, :))); 
            end
        end
        if ~have_raw
            save([raw_res '/responses.mat'],'responses');
        end
        
        %% SCRG
        a = S_out./C_out;a(isnan(a))=1;
        b = S_in./C_in;b(isnan(b))=1;
        scrg_tmp = sum(a)/sum(b);
        % 0 / 0 =NAN
        
        if ~exist('result/SCRG/','dir')
           mkdir('result/SCRG/'); 
        end
        if ~exist(['result/SCRG/' seqs{s} '_' detectors{dd} '_SCRG.mat'], 'file')
            SCRG(s, dd) = scrg_tmp;
            save(['result/SCRG/' seqs{s} '_' detectors{dd} '_SCRG.mat'], 'scrg_tmp');
        else
            load(['result/SCRG/' seqs{s} '_' detectors{dd} '_SCRG.mat']);
            SCRG(s, dd) = scrg_tmp;
        end
       %% BSF
        bsf_tmp = sum(C_in./C_out);
        if ~exist('result/BSF/','dir')
           mkdir('result/BSF/'); 
        end
        if ~exist(['result/BSF/' seqs{s} '_' detectors{dd} '_BSF.mat'], 'file')
            BSF(s, dd) = bsf_tmp;
            save(['result/BSF/' seqs{s} '_' detectors{dd} '_BSF.mat'], 'bsf_tmp');
        else
            load(['result/BSF/' seqs{s} '_' detectors{dd} '_BSF.mat']);
            BSF(s, dd) = bsf_tmp;
        end
       %% CG
        cg_tmp = sum(S_out./S_in);
        if ~exist('result/CG/','dir')
           mkdir('result/CG/'); 
        end
        if ~exist(['result/CG/' seqs{s} '_' detectors{dd} '_CG.mat'], 'file')
            CG(s, dd) = cg_tmp;
            save(['result/CG/' seqs{s} '_' detectors{dd} '_CG.mat'], 'cg_tmp');
        else
            load(['result/CG/' seqs{s} '_' detectors{dd} '_CG.mat']);
            CG(s, dd) = cg_tmp;
        end
        
    end
end
