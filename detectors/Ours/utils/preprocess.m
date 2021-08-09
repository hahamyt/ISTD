function img_ = preprocess(img)
%     min_val = min(img(:));
%     max_val = max(img(:));
%     img_ = (img - min_val)./(max_val - min_val);
    img_ = double(img);

    %% Image denoise
    se=strel('square',2);
    img_=imdilate(img_, se);
%     se=strel('square',2);
%     img_=imerode(img_,se);

    
    %% 
    op = fspecial('average', 9); % improved high boost filter
    Im = imfilter(double(img_), op, 'symmetric');
    Ihp = img_ - Im;
    Ihp(Ihp<0) = 0;
    ihbf = img_.*Ihp;
    img_ = ihbf;
    
%     subplot(132);
%     imagesc(Ihp);
end

