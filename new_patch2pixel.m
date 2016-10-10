function [input_patches, label_patches, deinterlaced_patches, eachCnt] = new_patch2pixel(frames)
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
    window = 5;
    padding = (window-1) / 2;
    deinterlaced_fields = padarray(deinterlaced_fields, [padding, padding], 'symmetric');
    
    %% Initialization
    input_patches = zeros(window, window, 1, 1);
    label_patches = zeros(1, 1, 1, 1);
    deinterlaced_patches = zeros(1, 1, 1, 1);
    count = 0;
    
    %% Generate data pacth
    for frameCnt = 1:cnt
        if frameCnt == 1
            input_full(:, :, 1) = deinterlaced_fields(:, :, frameCnt);
            input_full(:, :, 2) = deinterlaced_fields(:, :, frameCnt+1);
            input_full(:, :, 4) = deinterlaced_fields(:, :, frameCnt+1);
            input_full(:, :, 5) = deinterlaced_fields(:, :, frameCnt+2);
        elseif frameCnt == 2
            input_full(:, :, 1) = deinterlaced_fields(:, :, frameCnt);
            input_full(:, :, 2) = deinterlaced_fields(:, :, frameCnt-1);
            input_full(:, :, 4) = deinterlaced_fields(:, :, frameCnt+1);
            input_full(:, :, 5) = deinterlaced_fields(:, :, frameCnt+2);
        elseif frameCnt == cnt-1
            input_full(:, :, 1) = deinterlaced_fields(:, :, frameCnt-2);
            input_full(:, :, 2) = deinterlaced_fields(:, :, frameCnt-1);
            input_full(:, :, 4) = deinterlaced_fields(:, :, frameCnt+1);
            input_full(:, :, 5) = deinterlaced_fields(:, :, frameCnt);
        elseif frameCnt == cnt
            input_full(:, :, 1) = deinterlaced_fields(:, :, frameCnt-2);
            input_full(:, :, 2) = deinterlaced_fields(:, :, frameCnt-1);
            input_full(:, :, 4) = deinterlaced_fields(:, :, frameCnt-1);
            input_full(:, :, 5) = deinterlaced_fields(:, :, frameCnt);
        else
            input_full(:, :, 1) = deinterlaced_fields(:, :, frameCnt-2);
            input_full(:, :, 2) = deinterlaced_fields(:, :, frameCnt-1);
            input_full(:, :, 4) = deinterlaced_fields(:, :, frameCnt+1);
            input_full(:, :, 5) = deinterlaced_fields(:, :, frameCnt+2);
        end
        
        input_full(:, :, 3) = deinterlaced_fields(:, :, frameCnt);
        
        label_full = frames(:, :, frameCnt);
          
        % Odd frame => scan even row
        if mod(frameCnt, 2) == 1
            s_row = 2;
        % Even frame => scan odd row
        else
            s_row = 1;
        end
        
        mmax = 0;
        mcnt = 0;
        
        eachCnt = 0;
        for row = s_row:2:hei
            for col = 1:wid
                eachCnt = eachCnt + 1;
                count = count + 1;
                p_s_row = (row - 1) + padding;
                p_e_row = (row + 1) + padding;
                p_s_col = (col - padding) + padding;
                p_e_col = (col + padding) + padding;
                
                input_patches(1, :, :, count) = input_full(row+padding, p_s_col:p_e_col, 2);
                input_patches(2, :, :, count) = input_full(p_s_row, p_s_col:p_e_col, 3);
                input_patches(3, :, :, count) = input_full(p_e_row, p_s_col:p_e_col, 3);
                input_patches(4, :, :, count) = input_full(row+padding, p_s_col:p_e_col, 4);
                
                seq1 = sum((input_full(p_s_row, p_s_col:p_e_col, 1) - input_full(p_s_row, p_s_col:p_e_col, 3)) .^ 2) + ...
                       sum((input_full(p_e_row, p_s_col:p_e_col, 1) - input_full(p_e_row, p_s_col:p_e_col, 3)) .^ 2);
                seq1 = seq1 / (2*window);
                
                seq2 = sum((input_full(p_s_row, p_s_col:p_e_col, 5) - input_full(p_s_row, p_s_col:p_e_col, 3)) .^ 2) + ...
                       sum((input_full(p_e_row, p_s_col:p_e_col, 5) - input_full(p_e_row, p_s_col:p_e_col, 3)) .^ 2);
                seq2 = seq2 / (2*window);
                
                input_patches(5, 1, :, count) = seq1;
                input_patches(5, 2, :, count) = seq2;
                
                % Mod1
                %input_patches(5, 3, :, count) = input_full(row+padding, col+padding, 3);
                
                % Mod2
                %input_patches(5, 3, :, count) = input_full(row+padding-1, col+padding, 3);
                %input_patches(5, 4, :, count) = input_full(row+padding+1, col+padding, 3);
                
                %{
                lfm1 = seq1;
                lfm2 = seq1;
                lfm = (lfm1+lfm2) / 2;
                                
                thre = 0.08;
                           
                if lfm >= thre
                    input_patches(1, :, :, count) = input_full(row+padding, p_s_col:p_e_col, 3);
                    input_patches(4, :, :, count) = input_full(row+padding, p_s_col:p_e_col, 3);
                end
                %}
                
                
                %{
                if mmax < lfm
                    mmax = lfm;
                end
                
                mmax = mmax + lfm;
                mcnt = mcnt + 1;
                %}
           
                label_patches(:, :, :, count) = label_full(row, col, :);
                deinterlaced_patches(:, :, :, count) = input_full(row+padding, col+padding, 3);
            end
        end
        
        disp([frameCnt mmax/mcnt]);
    end
end