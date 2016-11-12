% Patch => Patch method
function [input_patches, label_patches, interlaced_patches, deinterlaced_patches, inv_mask_patches, eachCnt] = patch2patch(frames, window, stride, input_channels)
    [hei, wid, cnt] = size(frames);
    %stride = min(window);
    
    %% Get frames, interlaced_fields, inv_masks, deinterlaced_fields
    for frameCnt = 1:cnt
        frame = frames(:, :, frameCnt);
        [interlaced_field, inv_mask] = interlace(frame, mod(frameCnt, 2));
        deinterlaced_field = deinterlace(interlaced_field, mod(frameCnt, 2));
        
        interlaced_fields(:, :, frameCnt) = interlaced_field;
        deinterlaced_fields(:, :, frameCnt) = deinterlaced_field;
        inv_masks(:, :, frameCnt) = inv_mask;
    end
    
    % Temp
    deinterlaced_fields = deinterlace_video(frames, cnt);
    
    frames = im2double(frames);
    interlaced_fields = im2double(interlaced_fields);        
    deinterlaced_fields = im2double(deinterlaced_fields);
    inv_masks = im2double(inv_masks);
    
    %% Initialization
    input_patches = zeros(window(1), window(2), input_channels, 1);
    label_patches = zeros(window(1), window(2), 1, 1);
    interlaced_patches = zeros(window(1), window(2), 1, 1);
    deinterlaced_patches = zeros(window(1), window(2), 1, 1);
    inv_mask_patches = zeros(window(1), window(2), 1, 1);
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
                
            %% Generate patchs from each
        % Record each frame has how many patches
        eachCnt = 0;
        for row = 1:stride:hei
            s_row = row;
            if s_row + window(1) - 1 > hei
                s_row = hei - window(1) + 1;
            end

            for col = 1:stride:wid
                s_col = col;
                if s_col + window(2) - 1 > wid
                    s_col = wid - window(2) + 1;
                end
                
                eachCnt = eachCnt + 1;
                count = count + 1;

                input_patches(:, :, :, count) = input_full(s_row:s_row+window(1)-1, s_col:s_col+window(2)-1, :);
                label_patches(:, :, :, count) = label_full(s_row:s_row+window(1)-1, s_col:s_col+window(2)-1, :);
                interlaced_patches(:, :, :, count) = interlace_full(s_row:s_row+window(1)-1, s_col:s_col+window(2)-1, :);
                deinterlaced_patches(:, :, :, count) = deinterlace_full(s_row:s_row+window(1)-1, s_col:s_col+window(2)-1, :);
                inv_mask_patches(:, :, :, count) = inv_mask_full(s_row:s_row+window(1)-1, s_col:s_col+window(2)-1, :);
            end
        end
    end
end