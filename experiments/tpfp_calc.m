function [tp, fp] = tpfp_calc(gt, pred, w)
    tp = 0;     % True Positive: Right Positive
    fp = 0;     % False Positive: Wrong Positive
    [np, ~] = size(pred);
        
    if np ~= 0
        tmp = abs(pred-gt);
        [tp, ~] = size(find(tmp(:, 1)<=w&tmp(:, 2)<=w));
        fp = np - tp;
    end
end