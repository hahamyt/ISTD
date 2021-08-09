function [fpr, tpr, precision, recall, f1_score, accuracy] = detect_threshold(response, threshold, gt)
    %% This function is created for detecting targets under given threshold
    % response: detection results
    % threshold: response threshold
    % gt: ground truth

    response = imgaussfilt(response,2);
    local = false;      % true: local region; false: whole region
    error("You should set the ground-truth region parameter {w}, {d} manually.");
    w = 4;
    if local == true
        error("You should set the ground-truth region parameter {d} manually.");
        d = 30;
        [w, h] = size(response);
        response = response(max(gt(1)-d,1):min(gt(1)+d, w), max(gt(2)-d,1):min(gt(2)+d, h));
        gt = [d, d];
    end

    response=(response-min(response(:)))/(max(response(:))-min(response(:)));
    max_val = max(response(:));
    min_val = min(response(:)); 
    t = (max_val - min_val) * threshold;

    pos_idxs = find(response(:)>=t);
    neg_idxs = find(response(:)<t); 
   
    [row, col]=ind2sub(size(response), pos_idxs);
    pos_points = [row, col];

    [tp, fp] = tpfp_calc(gt, pos_points, w);

    [row, col]=ind2sub(size(response), neg_idxs);
    neg_points = [row, col];
    [tn, fn] = tnfn_calc(gt, neg_points, w);

    recall = tp/(tp+fn);
    precision = tp/(tp+fp);
    tpr = tp/(tp+fn);
    fpr = fp/(fp+tn);
    alpha = 1;
    f1_score = ((alpha+1)*precision*recall)/(alpha*precision+recall);
    accuracy = (tp+tn)/(tp+tn+fp+fn);
end