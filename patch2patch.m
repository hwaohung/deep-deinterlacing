% Patch => Patch method
function [input_patches, label_patches, deinterlaced_patches, eachCnt] = patch2patch(frames, window, stride, input_channels)
    [hei, wid, cnt] = size(frames);
    
    %% Get frames, interlaced_fields, inv_masks, deinterlaced_fields    
    deinterlaced_fields = self_validation(frames, 6, 1);
    
    frames = im2double(frames);
    deinterlaced_fields = im2double(deinterlaced_fields);
    
    %% Initialization   
    rows = (1:stride:hei-window(1)+1);
    if rows(end) ~= hei-window(1)+1
        rows(end+1) = hei-window(1)+1;
    end
    
    cols = (1:stride:wid-window(2)+1);
    if cols(end) ~= wid-window(2)+1
        cols(end+1) = wid-window(2)+1;
    end
    
    eachCnt = size(rows, 2) * size(cols, 2);
    input_patches = zeros((window(1)/2)*4+1, window(2), 1, cnt*eachCnt);
    label_patches = zeros(window(1)/2, window(2), 1, cnt*eachCnt);
    deinterlaced_patches = zeros(window(1)/2, window(2), 1, cnt*eachCnt);
    count = 0;
    
    %% Generate data pacth
    for frameCnt = 1:cnt
        curr = deinterlaced_fields(:, :, frameCnt);
        if frameCnt == 1
            prev = deinterlaced_fields(:, :, frameCnt+1);
            post = deinterlaced_fields(:, :, frameCnt+1);
        elseif frameCnt == cnt
            prev = deinterlaced_fields(:, :, frameCnt-1);
            post = deinterlaced_fields(:, :, frameCnt-1);
        else
            prev = deinterlaced_fields(:, :, frameCnt-1);
            post = deinterlaced_fields(:, :, frameCnt+1);
        end
        
        for row = rows
            odd_row_indexes = (row-1+1:2:row+window(1)-1);
            even_row_indexes = (row-1+2:2:row+window(1)-1);
            
            if odd_row_indexes(end) + 2 > hei
                spec_end = odd_row_indexes(end);
            else
                spec_end = odd_row_indexes(end) + 2;
            end
            
            if even_row_indexes(1) - 2 < 1
                spec_first = even_row_indexes(1);
            else
                spec_first = even_row_indexes(1) - 2;
            end
            
            for col = cols
                count = count + 1;
                               
                col_indexes = (col:col + window(2) - 1);
                
                if mod(frameCnt, 2)
                    input_patches(2:4:end, :, :, count) = prev(even_row_indexes, col_indexes);
                    input_patches(3:4:end, :, :, count) = curr(even_row_indexes, col_indexes);
                    input_patches(4:4:end, :, :, count) = post(even_row_indexes, col_indexes);
                    
                    input_patches(1:4:end-4, :, :, count) = curr(odd_row_indexes, col_indexes);
                    input_patches(end, :, :, count) = curr(spec_end, col_indexes);
                                                            
                    label_patches(:, :, :, count) = frames(even_row_indexes, col_indexes, frameCnt);
                    deinterlaced_patches(:, :, :, count) = curr(even_row_indexes, col_indexes);
                else
                    input_patches(2:4:end, :, :, count) = prev(odd_row_indexes, col_indexes);
                    input_patches(3:4:end, :, :, count) = curr(odd_row_indexes, col_indexes);
                    input_patches(4:4:end, :, :, count) = post(odd_row_indexes, col_indexes);
                    
                    input_patches(5:4:end, :, :, count) = curr(even_row_indexes, col_indexes);
                    input_patches(1, :, :, count) = curr(spec_first, col_indexes);
                                
                    label_patches(:, :, :, count) = frames(odd_row_indexes, col_indexes, frameCnt);
                    deinterlaced_patches(:, :, :, count) = curr(odd_row_indexes, col_indexes);
                end
            end            
        end
    end 
end