% Patch => Patch method
function [input_patches, label_patches, deinterlaced_patches, flags, eachCnt] = patch2patch(frames, window, stride)
    [hei, wid, ch, cnt] = size(frames);
    
    %% Get frames, interlaced_fields, inv_masks, deinterlaced_fields    
    %deinterlaced_fields = deinterlace(frames, 1);
    [deinterlaced_fields, prpo_fields] = deinterlace_video(frames, cnt);
    
    frames = im2double(frames);
    deinterlaced_fields = im2double(deinterlaced_fields);
    prpo_fields = im2double(prpo_fields);
    
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
    input_patches = zeros((window(1)/2)*4+1, window(2), 3, cnt*eachCnt);
    label_patches = zeros(window(1)/2, window(2), 3, cnt*eachCnt);
    deinterlaced_patches = zeros(window(1)/2, window(2), 3, cnt*eachCnt);
    flags = zeros(1, 1, 1, cnt*eachCnt);
    count = 0;
    
    %% Generate data pacth
    for frameCnt = 1:cnt
        curr = deinterlaced_fields(:, :, :, frameCnt);
        prpo_field = prpo_fields(:, :, :, frameCnt);
        
        if frameCnt == 1
            prev2 = deinterlaced_fields(:, :, :, frameCnt+2);
            prev1 = deinterlaced_fields(:, :, :, frameCnt+1);
            post1 = deinterlaced_fields(:, :, :, frameCnt+1);
            post2 = deinterlaced_fields(:, :, :, frameCnt+2);
        elseif frameCnt == 2
            prev2 = deinterlaced_fields(:, :, :, frameCnt+2);
            prev1 = deinterlaced_fields(:, :, :, frameCnt-1);
            post1 = deinterlaced_fields(:, :, :, frameCnt+1);
            post2 = deinterlaced_fields(:, :, :, frameCnt+2);
        elseif frameCnt == cnt-1
            prev2 = deinterlaced_fields(:, :, :, frameCnt-2);
            prev1 = deinterlaced_fields(:, :, :, frameCnt-1);
            post1 = deinterlaced_fields(:, :, :, frameCnt+1);
            post2 = deinterlaced_fields(:, :, :, frameCnt-2);
        elseif frameCnt == cnt
            prev2 = deinterlaced_fields(:, :, :, frameCnt-2);
            prev1 = deinterlaced_fields(:, :, :, frameCnt-1);
            post1 = deinterlaced_fields(:, :, :, frameCnt-1);
            post2 = deinterlaced_fields(:, :, :, frameCnt-2);
        else
            prev2 = deinterlaced_fields(:, :, :, frameCnt-2);
            prev1 = deinterlaced_fields(:, :, :, frameCnt-1);
            post1 = deinterlaced_fields(:, :, :, frameCnt+1);
            post2 = deinterlaced_fields(:, :, :, frameCnt+2);
        end
        
        %{
        t1 = rgb2gray(curr) * 255;
        t2 = rgb2gray(prev2) * 255;
        t3 = rgb2gray(post2) * 255;
        
        lfmt1 = (t1 - t2) .^ 2;
        lfmt2 = (t1 - t3) .^ 2;
        %}
        
        t1 = rgb2gray(prev1) * 255;
        t2 = rgb2gray(post1) * 255;
        
        lfmt1 = abs(t1 - t2);
        lfmt2 = lfmt1;
        
        if mod(frameCnt, 2)
            for row = rows
                odd_row_indexes = (row:2:row+window(1)-1);
                even_row_indexes = (row+1:2:row+window(1)-1);
            
                % odd scan bottom boundary
                if odd_row_indexes(end) + 2 > hei
                    spec_end = odd_row_indexes(end);
                else
                    spec_end = odd_row_indexes(end) + 2;
                end
            
                for col = cols
                    count = count + 1;
                    col_indexes = (col:col + window(2) - 1);
                       
                    %lfm1 = lfmt1(odd_row_indexes, col_indexes);
                    %lfm2 = lfmt2(odd_row_indexes, col_indexes);
                    lfm1 = lfmt1(even_row_indexes, col_indexes);
                    lfm2 = lfmt2(even_row_indexes, col_indexes);
                    lfm1 = mean(lfm1(:));
                    lfm2 = mean(lfm2(:));
                    flags(:, :, :, count) = (lfm1 + lfm2) / 2;
                    
                    input_patches(3:4:end, :, :, count) = curr(even_row_indexes, col_indexes, :);
                    
                    
                    if flags(:, :, :, count) < Var.T
                        input_patches(2:4:end, :, :, count) = prev1(even_row_indexes, col_indexes, :);                    
                        input_patches(4:4:end, :, :, count) = post1(even_row_indexes, col_indexes, :);
                    else
                        input_patches(2:4:end, :, :, count) = prpo_field(even_row_indexes-1, col_indexes, :);
                        input_patches(4:4:end, :, :, count) = prpo_field(even_row_indexes, col_indexes, :);
                    end
                                                            
                    input_patches(1:4:end-4, :, :, count) = curr(odd_row_indexes, col_indexes, :);
                    input_patches(end, :, :, count) = curr(spec_end, col_indexes, :);
                                                            
                    label_patches(:, :, :, count) = frames(even_row_indexes, col_indexes, :, frameCnt);
                    deinterlaced_patches(:, :, :, count) = curr(even_row_indexes, col_indexes, :);
                end
            end
        else
            for row = rows
                odd_row_indexes = (row:2:row+window(1)-1);
                even_row_indexes = (row+1:2:row+window(1)-1);
            
                % even scan top boundary
                if even_row_indexes(1) - 2 < 1
                    spec_first = even_row_indexes(1);
                else
                    spec_first = even_row_indexes(1) - 2;
                end
            
                for col = cols
                    count = count + 1;
                    col_indexes = (col:col + window(2) - 1);
                    
                    %lfm1 = lfmt1(even_row_indexes, col_indexes);
                    %lfm2 = lfmt2(even_row_indexes, col_indexes);
                    lfm1 = lfmt1(odd_row_indexes, col_indexes);
                    lfm2 = lfmt2(odd_row_indexes, col_indexes);
                    lfm1 = mean(lfm1(:));
                    lfm2 = mean(lfm2(:));
                    flags(:, :, :, count) = (lfm1 + lfm2) / 2;
                    
                    input_patches(3:4:end, :, :, count) = curr(odd_row_indexes, col_indexes, :);
                                        
                    if flags(:, :, :, count) < Var.T
                        input_patches(2:4:end, :, :, count) = prev1(odd_row_indexes, col_indexes, :);                    
                        input_patches(4:4:end, :, :, count) = post1(odd_row_indexes, col_indexes, :);
                    else
                        input_patches(2:4:end, :, :, count) = prpo_field(odd_row_indexes, col_indexes, :);
                        input_patches(4:4:end, :, :, count) = prpo_field(odd_row_indexes+1, col_indexes, :);
                    end
                    
                    input_patches(5:4:end, :, :, count) = curr(even_row_indexes, col_indexes, :);
                    input_patches(1, :, :, count) = curr(spec_first, col_indexes, :);
                                
                    label_patches(:, :, :, count) = frames(odd_row_indexes, col_indexes, :, frameCnt);
                    deinterlaced_patches(:, :, :, count) = curr(odd_row_indexes, col_indexes, :);
                end
            end
        end
    end 
end
