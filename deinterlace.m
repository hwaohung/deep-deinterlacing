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
    % ELA
    elseif method == 4
        field = im2double(field);
        image = field;
        padding = 1;
        field = padarray(field, [padding, padding], 'symmetric');
        
        p1 = [0, 1, 0; ...
              0, 0, 0; ...
              0, 0, -1];
          
        p2 = [1, 0, 0; ...
              0, 0, 0; ...
              0, -1, 0];
          
        q1 = [0, 0, 1; ...
              0, 0, 0; ...
              0, -1, 0];
        
        q2 = [0, 1, 0; ...
              0, 0, 0; ...
              -1, 0, 0];
          
        p = abs(conv2(field, p1, 'valid')) + abs(conv2(field, p2, 'valid'));
        q = abs(conv2(field, q1, 'valid')) + abs(conv2(field, q2, 'valid'));
                       
        c1 = [1, 0, 0; ...
              0, 0, 0; ...
              0, 0, -1];
          
        c2 = [0, 1, 0; ...
              0, 0, 0; ...
              0, -1, 0];
          
        c3 = [0, 0, 1; ...
              0, 0, 0; ...
              -1, 0, 0];
                
        c1 = abs(conv2(field, c1, 'valid'));
        c2 = abs(conv2(field, c2, 'valid'));
        c3 = abs(conv2(field, c3, 'valid'));
        
        if ~odd
            s_i = 1;
        else
            s_i = 2;
        end     
                
        for i = s_i:2:size(image, 1)
            for j = 1:size(image, 2)
                if p(i, j) > q(i, j)
                    if c3(i, j) >= c2(i, j)
                        k = 0;
                    else
                        k = -1;
                    end
                elseif p < q
                    if c2(i, j) >= c1(i, j)
                        k = 1;
                    else
                        k = 0;
                    end
                else
                    [t, k] = min([c3(i, j), c2(i, j), c1(i, j)]);
                    k = k-2;
                end
                                
                image(i, j) = (field(i+padding-1, j+padding-k) + field(i+padding+1, j+padding+k)) / 2;
            end
        end
        
        image = uint8(image * 255);
    end
end