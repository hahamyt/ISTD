%% Config
addpath('./experiments');
addpath('./utilities');
clear;clc;
root_dir = pwd;
%% Visulation or not
viz = true;
%% Necessary settings  
% detectors = { 'IPI','Ours', 'LEF', 'WSLCM', 'FKRW', 'ADDGD', 'HBMLCM', 'LIG', ...
%     'ADMD', 'PSTNN', 'MAXMEAN', 'TopHat'}; %
detectors = { 'LEF'}; %

% detectors = {'baseline','baseline_Huber', 'baseline_Huber_NN', 'baseline_Huber_NN_Prior'};
% detectors = {'IPI'};

% seqs = {'seq1', 'seq2', 'seq3','seq4', 'seq5', 'seq6'};
seqs = {'seq1'};

[~, num_seq] = size(seqs);
[~, num_det] = size(detectors);
%% Preload

%% Start detection sequences
fprintf('Detecting\n');
for dd = 1:num_det
    fprintf('Detector: %s\n', detectors{dd});
    for s = 1:num_seq
        
        fprintf('Seq: %s\n', seqs{s});
        d = dir(['./data/' seqs{s}]);
        nameCell = cell(length(d)-2,1);  % arrange the order of the seq
        for i = 3:length(d)
        %     disp(d(i).name)
            nameCell{i-2} = strcat(d(i).folder,'/', d(i).name);%d(i).name;
        end
        imglist = sort_nat(nameCell);
        responses = cell(length(imglist), 1);
        %% load the pregenerated raw responses: for faster evaluation
        raw_res = ['./result/' seqs{s} '/' detectors{dd} '/raw_responses'];
        if ~exist(raw_res, 'dir')
            mkdir(raw_res);
        end
        have_raw = 0;
        if exist([raw_res '/responses.mat'], 'file')
           load([raw_res '/responses.mat']);
           have_raw = 1;
        end
        
        for k=206:207%1:length(imglist)            
           %% Run: Collecting responses
            img=imread(imglist{k});
            if size(img, 3) == 3
                img = double(rgb2gray(img));
            end
            img = double(img); 
        if ~have_raw
            tic;
            funcName = ['response=run_' detectors{dd} '(img);'];
            cd(['./detectors/' detectors{dd}]);
            addpath(genpath('./'));
            eval(funcName);
            tt = toc;
            cd(root_dir);

            % preds
            response = gather(response);
            responses{k} = response;
            fprintf('%d/%d, time: %3s \n', k, length(imglist), tt);
        else
            response = responses{k};
            response=(response-min(response(:)))/(max(response(:))-min(response(:)));
        end
        clear response;

        if viz
            figure(1);
            subplot(121);
            imshow(img./255);
            subplot(122);
            imagesc(responses{k});
            axis off;
            pause(0.8);
            fprintf('%d/%d, \n', k, length(imglist));
        end
        end
        if ~have_raw
            save([raw_res '/responses.mat'],'responses');
        end
      
    end
end