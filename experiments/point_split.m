function [points, num_p] = point_split(pred)
    % elimate the zero elements
    pred(pred==0) = [];
    points = reshape(pred, 2, [])';
    [num_p, ~] = size(points);
end