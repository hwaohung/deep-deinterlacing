% Patch => Pixel method
% Note: window should be odd integer and greater equal than 3
function [input_patches, label_patches, eachCnt] = patch2pixel(frames, window, input_channels)
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