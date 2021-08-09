function [tn, fn] = tnfn_calc(gt, pred, w)
    tn = 0;     % True Negative: Right Negative
    fn = 0;     % False Negative: Wrong Negative
    [np, ~] = size(pred);
    if np ~= 0
        tmp = abs(pred-gt);
        [fn, ~] = size(find(tmp(:, 1)<=w&tmp(:, 2)<=w));
        tn = np - fn;
    end 
end