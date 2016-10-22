function [input_patches, label_patches, deinterlaced_patches, eachCnt] = new_patch2pixel(frames)
    [hei, wid, cnt] = size(frames);
    
    %% Prepare required data
    for frameCnt = 1:cnt
        %frame = frames(:, :, frameCnt);
        %[interlaced_field] = interlace(frame, mod(frameCnt, 2));
        %deinterlaced_fields(:, :, frameCnt) = deinterlace(interlaced_field, mod(frameCnt, 2));
        deinterlaced_fields(:, :, frameCnt) = deinterlace(frames(:, :, frameCnt), mod(frameCnt, 2));
    end

    frames = im2double(frames);
    deinterlaced_fields = im2double(deinterlaced_fields);
    
    % Do padding
    window = 3;
    padding = (window-1) / 2;
    deinterlaced_fields = padarray(deinterlaced_fields, [padding, padding], 'symmetric');
    
    %% Initialization
    input_patches = zeros(4, 4, 1, 1);
    %label_patches = zeros(1, 1, 1, 1);
    deinterlaced_patches = zeros(1, 1, 1, 1);
    count = 0;
    
    % Column major, external should tranpose
    tmp1(:, :, 1:2:cnt) = frames(2:2:hei, 1:wid, 1:2:cnt);
    tmp1(:, :, 2:2:cnt) = frames(1:2:hei, 1:wid, 2:2:cnt);
    tmp1 = reshape(tmp1, [wid, hei/2*cnt, 1]);
    tmp1 = transpose(tmp1);
    label_patches = reshape(tmp1, [1, 1, 1, size(tmp1(:), 1)]);   
    
    %% Generate data pacth
    padFrame = 2;
    deinterlaced_fields = cat(3, deinterlaced_fields(:, :, 1), deinterlaced_fields(:, :, 2), deinterlaced_fields, deinterlaced_fields(:, :, cnt-1), deinterlaced_fields(:, :, cnt));
    for frameCnt = 1:cnt
        input_full(:, :, 1) = deinterlaced_fields(:, :, frameCnt+padFrame-2);
        input_full(:, :, 2) = deinterlaced_fields(:, :, frameCnt+padFrame-1);
        input_full(:, :, 4) = deinterlaced_fields(:, :, frameCnt+padFrame+1);
        input_full(:, :, 5) = deinterlaced_fields(:, :, frameCnt+padFrame+2);
        
        input_full(:, :, 3) = deinterlaced_fields(:, :, frameCnt+padFrame);
        
        % Odd frame => scan even row
        if mod(frameCnt, 2) == 1
            s_row = 2;
        % Even frame => scan odd row
        else
            s_row = 1;
        end
                        

        % TODO: Use matrix to generate input_patches instead of loop        
        thr = 0.3;
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
                
                input_patches(1, 1:3, :, count) = input_full(row+padding, p_s_col:p_e_col, 2);
                input_patches(2, 1:3, :, count) = input_full(p_s_row, p_s_col:p_e_col, 3);
                input_patches(3, 1:3, :, count) = input_full(p_e_row, p_s_col:p_e_col, 3);
                input_patches(4, 1:3, :, count) = input_full(row+padding, p_s_col:p_e_col, 4);   
                                
                % Origin
                %{
                seq1 = sum((input_full(p_s_row, p_s_col:p_e_col, 1) - input_full(p_s_row, p_s_col:p_e_col, 3)) .^ 2) + ...
                       sum((input_full(p_e_row, p_s_col:p_e_col, 1) - input_full(p_e_row, p_s_col:p_e_col, 3)) .^ 2);
                seq1 = seq1 / (2*window);
                
                seq2 = sum((input_full(p_s_row, p_s_col:p_e_col, 5) - input_full(p_s_row, p_s_col:p_e_col, 3)) .^ 2) + ...
                       sum((input_full(p_e_row, p_s_col:p_e_col, 5) - input_full(p_e_row, p_s_col:p_e_col, 3)) .^ 2);
                seq2 = seq2 / (2*window);
                
                input_patches(1, 4, :, count) = seq1;
                input_patches(2, 4, :, count) = seq2;
                %}
                                
                % Method1
                %input_patches(1, 4, :, count) = seq1;
                %input_patches(2, 4, :, count) = seq2;
                %input_patches(3, 4, :, count) = input_full(row+padding, col+padding, 3);
                
                % Method2
                %input_patches(1, 4, :, count) = input_full(row+padding, col+padding, 1) - input_full(row+padding, col+padding, 3);
                %input_patches(2, 4, :, count) = input_full(row+padding, col+padding, 5) - input_full(row+padding, col+padding, 3);
                %input_patches(3, 4, :, count) = input_full(row+padding, col+padding, 3);
                
                % Method3
                %input_patches(1, 4, :, count) = abs(input_full(row+padding, col+padding, 1) - input_full(row+padding, col+padding, 3));
                %input_patches(2, 4, :, count) = abs(input_full(row+padding, col+padding, 5) - input_full(row+padding, col+padding, 3));
                %input_patches(3, 4, :, count) = input_full(row+padding, col+padding, 3);
                
                % Method4
                %input_patches(1, 4, :, count) = input_full(row+padding, col+padding, 1) - input_full(row+padding, col+padding, 3);
                %input_patches(2, 4, :, count) = input_full(row+padding, col+padding, 3) - input_full(row+padding, col+padding, 5);
                %input_patches(3, 4, :, count) = input_full(row+padding, col+padding, 3);
                
                % Method5
                input_patches(1, 4, :, count) = input_full(row+padding, col+padding, 1) - input_full(row+padding, col+padding, 3);
                input_patches(2, 4, :, count) = input_full(row+padding, col+padding, 5) - input_full(row+padding, col+padding, 3);
                input_patches(3, 4, :, count) = input_full(row+padding, col+padding, 3);
                input_patches(4, 4, :, count) = sum(abs(input_full(p_s_row, p_s_col:p_e_col, 2) - input_full(p_s_row, p_s_col:p_e_col, 4))) / window;
                
                seq1 = sum(abs(input_full(p_s_row, p_s_col:p_e_col, 2) - input_full(p_s_row, p_s_col:p_e_col, 4)));
                seq1 = seq1 / window;
                
                lfm1 = seq1;
                lfm2 = seq1;
                lfm = (lfm1+lfm2) / 2;             
                                
                % Analysis: Record all lfm for
                %{
                global index rec
                rec(index) = lfm;
                index = index + 1;
                %}
                
                % Test: According to the lfm change the input
                %{
                if lfm >= thre
                    input_patches(1, :, :, count) = input_full(row+padding, p_s_col:p_e_col, 3);
                    input_patches(4, :, :, count) = input_full(row+padding, p_s_col:p_e_col, 3);
                end
                %}
                
                %deinterlaced_patches(:, :, :, count) = input_full(row+padding, col+padding, 3);
                deinterlaced_patches(:, :, :, count) = lfm >= thr;
            end
        end
        
        disp([frameCnt mmax/mcnt]);
    end
end