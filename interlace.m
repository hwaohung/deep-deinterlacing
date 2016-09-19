% image: gray image with uint8
function [field, mask] = interlace(image, odd)
    % Only support 1 channel
    if size(image, 3) > 1
        image = rgb2gray(image);
    end
    
    if  ~strcmp(class(image), 'uint8')
        image = im2uint8(image);
    end
    
    imageSize = size(image);
    mask = uint8(zeros(imageSize(1), imageSize(2)));
    field = uint8(zeros(imageSize(1), imageSize(2)));
    
    % Remain odd field
    if odd
        mask(1:2:end, :) = 255;
        field(1:2:end, :)  = image(1:2:end, :);
    % Remain even field
    else
        mask(2:2:end, :) = 255;
        field(2:2:end, :) = image(2:2:end, :);
    end
    
    mask = abs(255 - mask);