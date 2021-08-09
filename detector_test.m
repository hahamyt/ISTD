clc;clear;
addpath('utilities');
%% This code is created for testing detectors
detector = 'Ours';

seq = 'seq6';
root_dir = pwd;
%% detector excuted
funcName = ['res=run_' detector '(img);'];

%% data path
d = dir(['./data/' seq]);
nameCell = cell(length(d)-2,1);  % arrange the order of the seq
for i = 3:length(d)
    nameCell{i-2} = strcat(d(i).folder,'/', d(i).name);%d(i).name;
end
imglist = sort_nat(nameCell);
gts = load(['data/groundturth/' seq '.txt']);
total_tt = 0;
%% Start
for k = 203:length(imglist)
    gt = gts(k, :);
    img=imread(imglist{k});
    if size(img, 3) == 3
        img = double(rgb2gray(img));
    end
    img = double(img);
    [w, h] = size(img);

    %% Run
    tic;
    
    cd(['./detectors/' detector]);
    addpath(genpath('./'));
    eval(funcName);
    tt = toc;
    cd(root_dir); 
    total_tt = total_tt + tt;
    fprintf([num2str(k) '/' num2str(length(imglist)) ', Time:' num2str(tt) '\n']);
    
%     res = imgaussfilt(res,4);
    maxval=max(max(res));
    
%     res(res~=maxval)=0;
    
    figure(1);
    set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w')
    subplot(131);
    imshow(img./255);
    title('Input');
    subplot(132);
    set(gca,'ytick',[]);
    set(gca,'xtick',[]);
    imagesc(res);
    title('Response');
    axis off;
    subplot(133);
    d=8;
    rres = res(max(gt(1)-d,1):min(gt(1)+d, w), max(gt(2)-d,1):min(gt(2)+d, h));
    imagesc(rres);
    title('Local Response Region');
    axis off;
    
    [x,y]=find(res==maxval);
    a = [mean(x),mean(y)]
    
    pause(0.1);
end

fprintf(['Mean Time:' num2str(total_tt/length(imglist)) '\n']);