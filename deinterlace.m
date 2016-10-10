% image: gray image with uint8
function [image] = deinterlace(field, odd)
    % Only support 1 channel
    if size(field, 3) > 1
        field = rgb2gray(field);
    end
    
    if  ~strcmp(class(field), 'uint8')
        field = im2uint8(field);
    end
    
    method = 1;
    
    if method == 1
        hdint = vision.Deinterlacer('Method', 'Linear interpolation', 'TransposedInput', false);
    
        if ~odd
            field(1:end-1, :) = field(2:end, :);
            image = step(hdint, field);
            image(2:end, :) = image(1:end-1, :);
        else
            image = step(hdint, field); 
        end
    elseif method == 2
        field = im2double(field);
        image = field;
        window = 3;
        padding = (window-1) / 2;
        mask = fspecial('gaussian', [2, window]);
        
        if ~odd
            field = field(2:2:end, :);
            field = padarray(field, [padding, 1], 'symmetric');
            interpolation = conv2(field, mask, 'valid');
            image(1:2:end, :) = interpolation(1:end-1, :);
        else
            field = field(1:2:end, :);
            field = padarray(field, [padding, 1], 'symmetric');
            interpolation = conv2(field, mask, 'valid');
            image(2:2:end, :) = interpolation(2:end, :);
        end
        
        image = uint8(image * 255);
    elseif method == 3
        if ~odd
            image = field;
            tmp = field(2:2:end, :);
            tmp = tmp(:, 2:2:end);
            tmp = imresize(tmp, 2);
            image(1:2:end, :) = tmp(1:2:end, :);
        else
            image = field;
            tmp = field(1:2:end, :);
            tmp = tmp(:, 1:2:end);
            tmp = imresize(tmp, 2);
            image(2:2:end, :) = tmp(2:2:end, :);
        end
    end
end