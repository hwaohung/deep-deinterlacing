% image: gray image with uint8
function [image] = deinterlace(field, odd)
    % Only support 1 channel
    if size(field, 3) > 1
        field = rgb2gray(field);
    end
    
    if  ~strcmp(class(field), 'uint8')
        field = im2uint8(field);
    end
    
    hdint = vision.Deinterlacer('Method', 'Linear interpolation', 'TransposedInput', false);
    
    if ~odd
        row1 = field(1, :);
        field(1:end-1, :) = field(2:end, :);
        image = step(hdint, field);
        image(2:end, :) = image(1:end-1, :);
        image(1, :) = row1(1, :);
    else
        image = step(hdint, field); 
    end
end