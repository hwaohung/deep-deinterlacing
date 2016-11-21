function [deinterlaced_frames] = self_validation(frames)
    %deinterlaced_frames = DeepTemp(frames, 100000, 1);
    %return;

    [hei, wid, cnt] = size(frames);
    
    methods = 6;
    
    field_map = uint8(zeros(hei/2, wid, cnt, methods));
    diff_map = zeros(hei/2 + 1, wid, cnt, methods);
    for method = 1:methods
        d1_frames = deinterlace(frames, 1, method);
        d2_frames = deinterlace(frames, 0, method);
        
        for i = 1:size(frames, 3)
            if mod(i, 2)
                field_map(:, :, i, method) = d1_frames(2:2:end, :, i);
                diff_map(1:end-1, :, i, method) = calc_diff_map(frames(1:2:end, :, i), d2_frames(1:2:end, :, i));
                diff_map(end, :, i, method) = diff_map(end-1, :, i, method);
            else
                field_map(:, :, i, method) = d1_frames(1:2:end, :, i);
                diff_map(2:end, :, i, method) = calc_diff_map(frames(2:2:end, :, i), d2_frames(2:2:end, :, i));
                diff_map(1, :, i, method) = diff_map(2, :, i, method);
            end
        end
    end
    
    ref_map = zeros(hei/2, wid, cnt, methods);
    window = [2, 5];
    mask = ones(window);
    diff_map = padarray(diff_map, [(window(1)-2)/2, (window(2)-1)/2], 'symmetric');
    
    for i = 1:size(diff_map, 4)
        for j = 1:size(diff_map, 3)
            ref_map(:, :, j, i) = conv2(diff_map(:, :, j, i), mask, 'valid');
        end
        
        for j = 1:size(ref_map, 3)
            if j == 1
                ref_map(:, :, j, i) = ref_map(:, :, j+1, i) * 0.25 + ref_map(:, :, j, i) + ref_map(:, :, j+1, i) * 0.25;
            elseif j == size(ref_map, 3)
                ref_map(:, :, j, i) = ref_map(:, :, j-1, i) * 0.25 + ref_map(:, :, j, i) + ref_map(:, :, j-1, i) * 0.25;
            else
                ref_map(:, :, j, i) = ref_map(:, :, j-1, i) * 0.25 + ref_map(:, :, j, i) + ref_map(:, :, j+1, i) * 0.25;
            end
        end
    end
    
    deinterlaced_frames = frames(:, :, :);
    % Get U
    [Y, U] = min(ref_map, [], 4);
    for i = 1:cnt
        if mod(i, 2)
            s_i = 2;
        else
            s_i = 1;
        end
        
        tmp = zeros(hei/2, wid);
        for row = 1:hei/2
            for col = 1:wid
                tmp(row, col) = field_map(row, col, i, U(row, col, i));
            end
        end
        
        deinterlaced_frames(s_i:2:end, :, i) = tmp;
    end
end

function [images] = deinterlace(fields, is_odd, method)
    if method == 1
        images = method1(fields, is_odd);
    elseif method == 2
        images = method2(fields, is_odd);
    elseif method == 3
        images = method3(fields, is_odd);
    elseif method == 4
        images = method4(fields, is_odd);
    elseif method == 5
        images = method5(fields, is_odd);
    elseif method == 6
        images = method6(fields, is_odd);
    elseif method == 7
        images = DeepTemp(fields, 100000, is_odd);
    end
end

function [images] = method1(fields, is_odd)
    images = zeros(size(fields), class(fields));
    
    fields = im2double(fields);
    mask = [0.5, 0, 0; 0, 0, 0.5];
    for i = 1:size(fields, 3)
        field = fields(:, :, i);
        image = field;
                
        if mod(i, 2) == is_odd
            field = field(1:2:end, :);
            field = padarray(field, [0, 1], 'symmetric');
            interpolation = conv2(field, mask, 'valid');
            image(2:2:end-1, :) = interpolation;
            image(end, :) = image(end-1, :);
        else                        
            field = field(2:2:end, :);
            field = padarray(field, [0, 1], 'symmetric');
            interpolation = conv2(field, mask, 'valid');
            image(3:2:end-1, :) = interpolation;
            image(1, :) = image(2, :);
        end
        
        images(:, :, i) = im2uint8(image);
    end
end

function [images] = method2(fields, is_odd)
    images = zeros(size(fields), class(fields));
    
    fields = im2double(fields);
    mask = [0.5; 0.5];
    for i = 1:size(fields, 3)
        field = fields(:, :, i);
        image = field;
                
        if mod(i, 2) == is_odd
            field = field(1:2:end, :);
            interpolation = conv2(field, mask, 'valid');
            image(2:2:end-1, :) = interpolation;
            image(end, :) = image(end-1, :);
        else                        
            field = field(2:2:end, :);
            interpolation = conv2(field, mask, 'valid');
            image(3:2:end-1, :) = interpolation;
            image(1, :) = image(2, :);
        end
        
        images(:, :, i) = im2uint8(image);
    end
end

function [images] = method3(fields, is_odd)
    images = zeros(size(fields), class(fields));
    
    fields = im2double(fields);
    mask = [0, 0, 0.5; 0.5, 0, 0];
    for i = 1:size(fields, 3)
        field = fields(:, :, i);
        image = field;
                
        if mod(i, 2) == is_odd
            field = field(1:2:end, :);
            field = padarray(field, [0, 1], 'symmetric');
            interpolation = conv2(field, mask, 'valid');
            image(2:2:end-1, :) = interpolation;
            image(end, :) = image(end-1, :);
        else                        
            field = field(2:2:end, :);
            field = padarray(field, [0, 1], 'symmetric');
            interpolation = conv2(field, mask, 'valid');
            image(3:2:end-1, :) = interpolation;
            image(1, :) = image(2, :);
        end
        
        images(:, :, i) = im2uint8(image);
    end
end

function [images] = method4(fields, is_odd)
    images = zeros(size(fields), class(fields));
    
    fields = im2double(fields);
    window = 3;
    padding = (window-1) / 2;
    mask = [0.0533, 0.3935, 0.0533;
            0.0533, 0.3935, 0.0533];
    
    for i = 1:size(fields, 3)
        field = fields(:, :, i);
        image = field;
                
        if mod(i, 2) == is_odd
            field = field(1:2:end, :);
            field = padarray(field, [1, padding], 'symmetric');
            interpolation = conv2(field, mask, 'valid');
            image(2:2:end, :) = interpolation(2:end, :);
        else                        
            field = field(2:2:end, :);
            field = padarray(field, [1, padding], 'symmetric');
            interpolation = conv2(field, mask, 'valid');
            image(1:2:end, :) = interpolation(1:end-1, :);
        end
        
        images(:, :, i) = im2uint8(image);
    end
end

function [images] = method5(fields, is_odd)
    images = zeros(size(fields), class(fields));
    
    fields = im2double(fields);
    for i = 1:size(fields, 3)
        field = fields(:, :, i);
        
        % Bilinear interpolation
        if mod(i, 2) == is_odd
            image = field;
            tmp = field(1:2:end, :);
            tmp = imresize(tmp, size(field), 'bilinear');
            image(2:2:end, :) = tmp(2:2:end, :);
        else            
            image = field;
            tmp = field(2:2:end, :);
            tmp = imresize(tmp, size(field), 'bilinear');
            image(1:2:end, :) = tmp(1:2:end, :);
        end        
        
        %{
        hdint = vision.Deinterlacer('Method', 'Vertical temporal median filtering', 'TransposedInput', false);
            
        if mod(i, 2) == is_odd
            image = step(hdint, field);
        else
            field(1:end-1, :) = field(2:end, :);
            image = step(hdint, field);
            image(2:end, :) = image(1:end-1, :);
        end
        %}
        
        images(:, :, i) = im2uint8(image);
    end
end

function [images] = method6(fields, is_odd)    
    c = size(fields, 3);
    images = zeros(size(fields), class(fields));
    %{
    for i = c:-1:2
        if mod(i, 2) == is_odd
            images(1:2:end, :, i) = fields(1:2:end, :, i);
            images(2:2:end, :, i) = fields(2:2:end, :, i-1);
        else
            images(2:2:end, :, i) = fields(2:2:end, :, i);
            images(1:2:end, :, i) = fields(1:2:end, :, i-1);
        end
    end
    
    if mod(1, 2) == is_odd
        images(1:2:end, :, 1) = fields(1:2:end, :, 1);
        images(2:2:end, :, 1) = fields(2:2:end, :, 2);
    else
        images(2:2:end, :, 1) = fields(2:2:end, :, 1);
        images(1:2:end, :, 1) =  fields(1:2:end, :, 2);
    end
    %}
    
    for i = 1:c-1
        if mod(i, 2) == is_odd
            images(1:2:end, :, i) = fields(1:2:end, :, i);
            images(2:2:end, :, i) = fields(2:2:end, :, i+1);
        else
            images(2:2:end, :, i) = fields(2:2:end, :, i);
            images(1:2:end, :, i) = fields(1:2:end, :, i+1);
        end
    end
    
    if mod(c, 2) == is_odd
        images(1:2:end, :, c) = fields(1:2:end, :, c);
        images(2:2:end, :, c) = fields(2:2:end, :, c-1);
    else
        images(2:2:end, :, c) = fields(2:2:end, :, c);
        images(1:2:end, :, c) =  fields(1:2:end, :, c-1);
    end
    
    %{
    for i = 2:c-1
        if mod(i, 2) == is_odd
            images(1:2:end, :, i) = fields(1:2:end, :, i);
            
            tmp = fields(1:2:end, :, i);
            
            tmp1 = method2(fields(:, :, i-1), 0); 
            tmp2 = method2(fields(:, :, i+1), 0);   
            tmp1 = sum(sum(abs(tmp-tmp1(1:2:end, :) .^ 2)));
            tmp2 = sum(sum(abs(tmp-tmp2(1:2:end, :) .^ 2)));
            
            if tmp1 > tmp2
                images(2:2:end, :, i) = fields(2:2:end, :, i+1);
            else
                images(2:2:end, :, i) = fields(2:2:end, :, i-1);
            end
        else
            images(2:2:end, :, i) = fields(2:2:end, :, i);
            
            tmp = fields(2:2:end, :, i);
            
            tmp1 = method2(fields(:, :, i-1), 1); 
            tmp2 = method2(fields(:, :, i+1), 1);   
            tmp1 = sum(sum(abs(tmp-tmp1(2:2:end, :) .^ 2)));
            tmp2 = sum(sum(abs(tmp-tmp2(2:2:end, :) .^ 2)));
            
            if tmp1 > tmp2
                images(1:2:end, :, i) = fields(1:2:end, :, i+1);
            else
                images(1:2:end, :, i) = fields(1:2:end, :, i-1);
            end
        end
    end
    
    if mod(c, 2) == is_odd
        images(1:2:end, :, c) = fields(1:2:end, :, c);
        images(2:2:end, :, c) = fields(2:2:end, :, c-1);
    else
        images(2:2:end, :, c) = fields(2:2:end, :, c);
        images(1:2:end, :, c) =  fields(1:2:end, :, c-1);
    end
    
    if mod(c, 2) == is_odd
        images(1:2:end, :, c) = fields(1:2:end, :, c);
        images(2:2:end, :, c) = fields(2:2:end, :, c-1);
    else
        images(2:2:end, :, c) = fields(2:2:end, :, c);
        images(1:2:end, :, c) =  fields(1:2:end, :, c-1);
    end
    %}
end

function [map] = calc_diff_map(frame1, frame2)
    map = abs(frame1 - frame2);
end