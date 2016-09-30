% Patch => Pixel method
% Note: window should be odd integer and greater equal than 3
function [input_patches, label_patches, eachCnt] = prepare_data(frames, window, input_channels)
    [hei, wid, cnt] = size(frames);
    
    %% Prepare required data
    for frameCnt = 1:cnt
        frame = frames(:, :, frameCnt);
        [interlaced_field] = interlace(frame, mod(frameCnt, 2));
        deinterlaced_fields(:, :, frameCnt) = deinterlace(interlaced_field, mod(frameCnt, 2));
    end

    frames = im2double(frames);
    deinterlaced_fields = im2double(deinterlaced_fields);
    
    % Do padding
    padding = (window-1) / 2;
    deinterlaced_fields = padarray(deinterlaced_fields, [padding, padding], 'symmetric');
    
    %% Initialization
    input_patches = zeros(window, window, input_channels, 1);
    label_patches = zeros(1, 1, 1, 1);
    count = 0;
    
    %% Generate data pacth
    for frameCnt = 1:cnt
        if input_channels == 3
            p_hei = hei + 2*padding;
            p_wid = wid + 2*padding;
            % Get prev, post field
            if frameCnt == 1
                input_full = reshape([deinterlaced_fields(:, :, frameCnt), ...
                                      deinterlaced_fields(:, :, frameCnt), ...
                                      deinterlaced_fields(:, :, frameCnt+1)], p_hei, p_wid, 3);
            elseif frameCnt == cnt
                input_full = reshape([deinterlaced_fields(:, :, frameCnt-1), ...
                                      deinterlaced_fields(:, :, frameCnt), ...
                                      deinterlaced_fields(:, :, frameCnt)], p_hei, p_wid, 3);
            else
                input_full = reshape([deinterlaced_fields(:, :, frameCnt-1), ...
                                      deinterlaced_fields(:, :, frameCnt), ...
                                      deinterlaced_fields(:, :, frameCnt+1)], p_hei, p_wid, 3);
            end
        elseif input_channels == 1
            input_full = deinterlaced_fields(:, :, frameCnt);
        end
        
        label_full = frames(:, :, frameCnt);
        
        % Odd frame => scan even row
        if mod(frameCnt, 2) == 1
            s_row = 2;
        % Even frame => scan odd row
        else
            s_row = 1;
        end
        
        eachCnt = 0;
        for row = s_row:2:hei
            for col = 1:wid
                eachCnt = eachCnt + 1;
                count = count + 1;
                p_s_row = (row - padding) + padding;
                p_e_row = (row + padding) + padding;
                p_s_col = (col - padding) + padding;
                p_e_col = (col + padding) + padding;
                input_patches(:, :, :, count) = input_full(p_s_row:p_e_row, p_s_col:p_e_col, :);
                label_patches(:, :, :, count) = label_full(row, col, :);
            end
        end
    end
end

% Patch => Patch method
%{
function [input_patchs, label_patchs, interlaced_patchs, deinterlaced_patchs, inv_mask_patchs, eachCnt] = prepare_data(frames, input_size, label_size, stride, input_channels)
    [hei, wid, cnt] = size(frames);
    %% Get frames, interlaced_fields, inv_masks, deinterlaced_fields
    for frameCnt = 1:cnt
        frame = frames(:, :, frameCnt);
        [interlaced_field, inv_mask] = interlace(frame, mod(frameCnt, 2));
        deinterlaced_field = deinterlace(interlaced_field, mod(frameCnt, 2));
        
        interlaced_fields(:, :, frameCnt) = interlaced_field;
        deinterlaced_fields(:, :, frameCnt) = deinterlaced_field;
        inv_masks(:, :, frameCnt) = inv_mask;
    end
    
    frames = im2double(frames);
    interlaced_fields = im2double(interlaced_fields);        
    deinterlaced_fields = im2double(deinterlaced_fields);
    inv_masks = im2double(inv_masks);
    
    %% Initialization
    input_patchs = zeros(input_size(1), input_size(2), input_channels, 1);
    label_patchs = zeros(label_size(1), label_size(2), 1, 1);
    interlaced_patchs = zeros(input_size(1), input_size(2), 1, 1);
    deinterlaced_patchs = zeros(input_size(1), input_size(2), 1, 1);
    inv_mask_patchs = zeros(input_size(1), input_size(2), 1, 1);
    count = 0;
    
    %% Generate data pacth
    for frameCnt = 1:cnt
        if input_channels == 3
            % Get prev, post field
            if frameCnt == 1
                prev = deinterlaced_fields(:, :, frameCnt);
                post = deinterlaced_fields(:, :, frameCnt+1);
            elseif frameCnt == cnt
                prev = deinterlaced_fields(:, :, frameCnt-1);
                post = deinterlaced_fields(:, :, frameCnt);
            else
                prev = deinterlaced_fields(:, :, frameCnt-1);
                post = deinterlaced_fields(:, :, frameCnt+1);
            end
            
            input_full = reshape([prev, deinterlaced_fields(:, :, frameCnt), post], hei, wid, 3);
        elseif input_channels == 1
            input_full = deinterlaced_fields(:, :, frameCnt);
        end
        
        label_full = frames(:, :, frameCnt);
        interlace_full = interlaced_fields(:, :, frameCnt);
        deinterlace_full = deinterlaced_fields(:, :, frameCnt);
        inv_mask_full = inv_masks(:, :, frameCnt);
                                
        % Test code for check image is ok
        %{
        if frameCnt == 2
            figure(1), imshow(input_full); title('Input Image');
            figure(2), imshow(label_full); title('Label Image');
            figure(3), imshow(interlace_full); title('Interlace Image');
            figure(4), imshow(deinterlace_full); title('De-interlace Image');
            figure(5), imshow(inv_mask_full); title('Mask Image');
            pause;
        end
        %}
                
        %% Generate patchs from each
        % Record each frame has how many patches
        eachCnt = 0;
        for row = 1:stride:hei
            s_row = row;
            if s_row + input_size(1) - 1 > hei
                s_row = hei - input_size(1) + 1;
            end

            for col = 1:stride:wid
                s_col = col;
                if s_col + input_size(2) - 1 > wid
                    s_col = wid - input_size(2) + 1;
                end
                
                eachCnt = eachCnt + 1;
                count = count + 1;

                input_patchs(:, :, :, count) = input_full(s_row:s_row+input_size(1)-1, s_col:s_col+input_size(2)-1, :);
                label_patchs(:, :, :, count) = label_full(s_row:s_row+label_size(1)-1, s_col:s_col+label_size(2)-1, :);
                interlaced_patchs(:, :, :, count) = interlace_full(s_row:s_row+input_size(1)-1, s_col:s_col+input_size(2)-1, :);
                deinterlaced_patchs(:, :, :, count) = deinterlace_full(s_row:s_row+input_size(1)-1, s_col:s_col+input_size(2)-1, :);
                inv_mask_patchs(:, :, :, count) = inv_mask_full(s_row:s_row+input_size(1)-1, s_col:s_col+input_size(2)-1, :);
            end
        end
    end
end
%}