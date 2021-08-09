%% Config
addpath('./experiments');
addpath('./utilities');
clear;clc;
root_dir = pwd;
%% Visulation or not
viz = false;
%% Necessary settings
detectors = { 'IPI','Ours', 'LEF', 'WSLCM', 'FKRW', 'ADDGD', 'HBMLCM', 'LIG', ...
    'ADMD', 'PSTNN', 'MAXMEAN', 'TopHat'}; %

seqs = {'seq1', 'seq2', 'seq3','seq4', 'seq5', 'seq6'};
thresholds = 0:0.01:1;

[~, num_seq] = size(seqs);
[~, num_det] = size(detectors);
%% ROC computing
num_thres = length(thresholds);
fpr = zeros(num_det, num_seq, num_thres);
tpr = zeros(num_det, num_seq, num_thres);
precision = zeros(num_det, num_seq, num_thres);
recall = zeros(num_det, num_seq, num_thres);
f1score = zeros(num_det, num_seq, num_thres);
accuracy = zeros(num_det, num_seq, num_thres);
%% Preload
%% ROC
tmp_roc_path = 'result/ROC';

%% Start detection sequences
fprintf('Detecting\n');
for dd = 1:num_det  
    fprintf('Detector: %s\n', detectors{dd});
    for s = 1:num_seq
        dfpr = zeros(1, num_thres);
        dtpr = zeros(1, num_thres);
        dprecision = zeros(1, num_thres);
        drecall = zeros(1, num_thres);
        df1score = zeros(1, num_thres);
        daccuracy = zeros(1, num_thres);

        if exist([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_dfpr.mat'], 'file')
            load([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_dfpr.mat']);fpr(dd,s,:)=dfpr(s,:);
            load([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_dtpr.mat']);tpr(dd,s,:)=dtpr(s,:);
            load([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_dprecision.mat']);precision(dd,s,:)=dprecision(s,:);
            load([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_drecall.mat']);recall(dd,s,:)=drecall(s,:);
            load([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_df1score.mat']);f1score(dd,s,:)=df1score(s,:);
            load([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_daccuracy.mat']);accuracy(dd,s,:)=daccuracy(s,:);
            continue;
        end
        %% images and responses generation
        gts = load(['data/groundturth/' num2str(seqs{s}) '.txt']);
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
            img=imread(imglist{k});
            if size(img, 3) == 3
                img = double(rgb2gray(img));
            end
            img = double(img);
           %% Run: Collecting responses
            if ~have_raw
                tic;
                funcName = ['response=run_' detectors{dd} '(img);'];
                cd(['./detectors/' detectors{dd}]);
                if k == 1
                    addpath(genpath('./'));
                end
                eval(funcName);
                cd(root_dir);

                % preds
                response = gather(response);
                responses{k} = response;
                fprintf('%d/%d, time: %3s \n', k, length(imglist), toc);
            else
                response = responses{k};
            end
            
            if viz
                subplot(121);
                imshow(img./255);
                subplot(122);
                imagesc(responses{k});
%                 imagesc(squeeze(responses(k, :, :))); 
            end
        end
        if ~have_raw
            save([raw_res '/responses.mat'],'responses');
        end        
       
       %% ROC
        % Calculate all thresholds under the same responses
        tic;
        for t = 1:num_thres
            threshold = thresholds(t);
            nn = length(imglist);
            fpr_tmp = zeros(nn, 1);tpr_tmp = zeros(nn, 1);
            precision_tmp = zeros(nn, 1);recall_tmp = zeros(nn, 1);
            f1score_tmp = zeros(nn, 1);accuracy_tmp = zeros(nn, 1);
%                 tic;
            for rr=1:nn
%                     disp(rr);
                [fpr_m, tpr_m, p, r, f1s, acc] = detect_threshold(squeeze(responses{rr}), threshold, gts(rr, :));
                fpr_tmp(rr,:) = fpr_m; %FP/(FP+TN);%
                tpr_tmp(rr,:) = tpr_m; %TP/(TP+FN);%
                precision_tmp(rr,:)= p; %TP/(TP+FP);%
                recall_tmp(rr,:)=r;%TP/(TP+FN);%
                f1score_tmp(rr,:)=f1s;
                accuracy_tmp(rr, :)= acc;
            end
            fpr_tmp = fpr_tmp(206:207,:);tpr_tmp = tpr_tmp(206:207,:);
            precision_tmp = precision_tmp(206:207,:);recall_tmp = recall_tmp(206:207,:);
            fpr_tmp(isnan(fpr_tmp))=[];dfpr(1, t) = mean(fpr_tmp(:));
            tpr_tmp(isnan(tpr_tmp))=[];dtpr(1, t) = mean(tpr_tmp(:));
            precision_tmp(isnan(precision_tmp))=[];dprecision(1, t) = mean(precision_tmp(:));
            recall_tmp(isnan(recall_tmp))=[];drecall(1, t) = mean(recall_tmp(:));
            f1score_tmp(isnan(f1score_tmp))=[];df1score(1, t) = mean(f1score_tmp(:));
            accuracy_tmp(isnan(accuracy_tmp))=[];daccuracy(1, t) = mean(accuracy_tmp(:));
%                 fprintf('Threshold: %f, Time: %s\n', threshold, toc);
        end
        dfpr(isnan(dfpr))=0;fpr(dd,s,:)=dfpr;
        dtpr(isnan(dtpr))=0;tpr(dd,s,:)=dtpr;
        dprecision(isnan(dprecision))=0;precision(dd,s,:)=dprecision;
        drecall(isnan(drecall))=0;recall(dd,s,:)=drecall;
        df1score(isnan(df1score))=0;f1score(dd,s,:)=df1score;
        daccuracy(isnan(daccuracy))=0;accuracy(dd, s, :)=daccuracy;
        if ~exist([tmp_roc_path '/' seqs{s}], 'dir')
            mkdir([tmp_roc_path '/' seqs{s}]);
        end
        
        if ~exist([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_dfpr.mat'], 'file')
            save([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_dfpr.mat'],'dfpr');
            save([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_dtpr.mat'],'dtpr'); 
            save([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_dprecision.mat'],'dprecision');
            save([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_drecall.mat'],'drecall');
            save([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_df1score.mat'],'df1score');
            save([tmp_roc_path  '/' seqs{s} '/' detectors{dd} '_daccuracy.mat'],'daccuracy');
        end
        fprintf('Seq: %s, Time: %s\n', seqs{s}, toc);
    end
end

fprintf('Plotting\n');
% Generating ROC plots
roc_plot(fpr, tpr, precision, recall, f1score, accuracy, detectors, seqs, length(thresholds));