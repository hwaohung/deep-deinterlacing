% Patch => Patch method
function [input_patches, label_patches, deinterlaced_patches, eachCnt] = patch2patch(frames, window, stride)
    [hei, wid, ch, cnt] = size(frames);
    
    %% Get frames, interlaced_fields, inv_masks, deinterlaced_fields    
    %deinterlaced_fields = deinterlace(frames, 1);
    deinterlaced_fields = deinterlace_video(frames, cnt);
    
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
    input_patches = zeros((window(1)/2)*4+1, window(2), 3, cnt*eachCnt);
    label_patches = zeros(window(1)/2, window(2), 3, cnt*eachCnt);
    deinterlaced_patches = zeros(window(1)/2, window(2), 3, cnt*eachCnt);
    count = 0;
    
    %% Generate data pacth
    for frameCnt = 1:cnt
        curr = deinterlaced_fields(:, :, :, frameCnt);
        if frameCnt == 1
            prev = deinterlaced_fields(:, :, :, frameCnt+1);
            post = deinterlaced_fields(:, :, :, frameCnt+1);
        elseif frameCnt == cnt
            prev = deinterlaced_fields(:, :, :, frameCnt-1);
            post = deinterlaced_fields(:, :, :, frameCnt-1);
        else
            prev = deinterlaced_fields(:, :, :, frameCnt-1);
            post = deinterlaced_fields(:, :, :, frameCnt+1);
        end
        
        %{
        tic;
        [new_prev, new_post] = refine_for_input(curr, prev, post, mod(frameCnt, 2));
        disp(toc);       
        
        if mod(frameCnt, 2)
            indexes = 1:2:size(curr, 1);
        else
            indexes = 2:2:size(curr, 1);
        end
                
        figure(1);
        a = new_post(:, :, :);
        a(indexes, :, :) = curr(indexes, :, :);
        imshow(a);
        figure(2);
        post(indexes, :, :) = curr(indexes, :, :);
        imshow(post);
        
        padding = 2;
        if mod(frameCnt, 2)
            row_indexes = 2+padding:2:size(curr, 1)-padding;
        else
            row_indexes = 1+padding:2:size(curr, 1)-padding;
        end
        
        for row = row_indexes
            for col = 1+padding:size(curr, 2)-padding
                if abs(prev(row, col, :) - post(row, col, :)) > 0.0340
                    prev(row, col, :) = new_prev(row, col, :);
                    post(row, col, :) = new_post(row, col, :);
                end
            end
        end
        %}
        
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
                    
                    input_patches(2:4:end, :, :, count) = prev(even_row_indexes, col_indexes, :);
                    input_patches(3:4:end, :, :, count) = curr(even_row_indexes, col_indexes, :);
                    input_patches(4:4:end, :, :, count) = post(even_row_indexes, col_indexes, :);
                    
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
                    
                    input_patches(2:4:end, :, :, count) = prev(odd_row_indexes, col_indexes, :);
                    input_patches(3:4:end, :, :, count) = curr(odd_row_indexes, col_indexes, :);
                    input_patches(4:4:end, :, :, count) = post(odd_row_indexes, col_indexes, :);
                    
                    input_patches(5:4:end, :, :, count) = curr(even_row_indexes, col_indexes, :);
                    input_patches(1, :, :, count) = curr(spec_first, col_indexes, :);
                                
                    label_patches(:, :, :, count) = frames(odd_row_indexes, col_indexes, :, frameCnt);
                    deinterlaced_patches(:, :, :, count) = curr(odd_row_indexes, col_indexes, :);
                end
            end
        end
    end 
end

% TODO:
% 2. Match function only for interlaced field
function [new_prev, new_post] = refine_for_input(curr, prev, post, is_odd)
    matcher = vision.TemplateMatcher('ROIInputPort', true);

    block = [9, 9];
    % padding must be even
    window = [5, 5];
    padding = (window(1)-1) / 2;

    new_prev = prev(:, :, :);
    new_post = post(:, :, :);
    
    % Index of curr deinterlaced field
    if is_odd
        row_indexes = 2+padding:2:size(curr, 1)-padding;
    else
        row_indexes = 1+padding:2:size(curr, 1)-padding;
    end
           
    % TODO: Process outer
    for row = row_indexes
        for col = 1+padding:size(curr, 2)-padding
            m_patch = curr(row-padding:row+padding, col-padding:col+padding);
            % TODO: Change the function only scan available row.
            loc = step(matcher, prev, m_patch, [col-(block(2)-1)/2, row-(block(1)-1)/2, block(2), block(1)]);
            new_prev(row, col, :) = prev(loc(1, 2), loc(1, 1), :);
            
            loc = step(matcher, post, m_patch, [col-(block(2)-1)/2, row-(block(1)-1)/2, block(2), block(1)]);
            new_post(row, col, :) = post(loc(1, 2), loc(1, 1), :);
        end
    end
end

function [loc] = finder(image, template, roi)
    %loc(1, 1);
    %loc(1, 2);
    
    block = [9, 9];
    % padding must be even
    window = [5, 5];
    padding = (window(1)-1) / 2;

    for row = row_indexes
        for col = col_indexes
            %image() = image
        end
    end
end