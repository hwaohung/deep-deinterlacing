function [im_hs, im_h_dsns, im_fusions, running_time] = DeepTemp(frames, iter, is_odd, spec)
    use_gpu = 1;
    input_channels = 3;
    
    % Set caffe mode
    if use_gpu
        caffe.set_mode_gpu();
        % use the first gpu in this demo
        gpu_id = 0;
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end
    
    tic; 
    [im_hs, im_h_dsns] = patch_deinterlace(frames, input_channels, iter, is_odd, spec);
    running_time = toc;
    
    im_hs = im2uint8(im_hs);
    im_h_dsns = im2uint8(im_h_dsns);
    im_fusions = (im_hs * 0.7) + (im_h_dsns * 0.3);

    % Reset caffe
    caffe.reset_all();
end

function [im_hs, im_h_dsns] = patch_deinterlace(frames, input_channels, iter, is_odd, spec)
    isPatch = 0;

    if isPatch
        % Patch size
        h = 16;
        w = 16;
        stride = min([h w]);
    else
        [h w c] = size(frames);
        stride = max([h w]);
    end
      
    model_dir = 'models/';
    
    net_model1 = [model_dir 'patch/DeepDeinterlacing_mat31.prototxt'];
    net_model2 = [model_dir 'patch/DeepDeinterlacing_mat11.prototxt'];
    
    net_weights1 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(1).caffemodel'];
    net_weights2 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(2).caffemodel'];

    if ~exist(net_weights1, 'file') || ~exist(net_weights2, 'file')
        error('Please check caffemodel is exist or not.');
    end
    
    if spec == 1
        net = caffe.Net(net_model1, net_weights1, 'test');
    else
        net = caffe.Net(net_model2, net_weights2, 'test');
    end

    input_channels = 1;
    [input_patches, label_patches, deinterlaced_patches, eachCnt] = patch2patch(frames, [h w], stride, input_channels, is_odd);
    
    init_net(net, h, w, input_channels, eachCnt);
    
    [hei, wid, cnt] = size(frames);
    rows = (1:stride:hei-h+1);
    if rows(end) ~= hei-h+1
        rows(end+1) = hei-h+1;
    end
    
    cols = (1:stride:wid-w+1);
    if cols(end) ~= wid-w+1
        cols(end+1) = wid-w+1;
    end
    
    im_hs = im2double(frames);
    im_h_dsns = im2double(frames);
    for i = 1:size(input_patches, 4)/eachCnt
        [im_h_patches1, im_h_dsn_patches1] = predict_patches(net, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                                             deinterlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt));
                                                                 
        j = 0;
        for row = rows
            if mod(i, 2)
                row_indexes = (row-1+2:2:row+h-1);
            else
                row_indexes = (row-1+1:2:row+h-1);
            end
            
            for col = cols
                j = j + 1;
                im_hs(row_indexes, col:col+w-1, i) = im_h_patches1(:, :, j);
                im_h_dsns(row_indexes, col:col+w-1, i) = im_h_dsn_patches1(:, :, j);
            end
        end
    end
end

function [] = init_net(net, h, w, input_channels, eachCnt)
    % Reshape blobs
    net.blobs('input').reshape([(h/2)*4+1 w input_channels eachCnt]);
    net.blobs('label').reshape([h/2 w 1 eachCnt]);
    net.blobs('deinterlace').reshape([h/2 w 1 eachCnt]);
    net.reshape();
end

% Patch => Patch method
function [input_patches, label_patches, deinterlaced_patches, eachCnt] = patch2patch(frames, window, stride, input_channels, is_odd)
    [hei, wid, cnt] = size(frames);
    
    %deinterlaced_fields = self_validation(frames, 6, is_odd);
    deinterlaced_fields = deinterlace(frames, 1);
    
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
        
        %{
        tic;
        [new_prev, new_post] = refine_for_input(curr, prev, post, mod(frameCnt, 2));
        disp(toc);
        
        padding = 2;
        if mod(frameCnt, 2) == is_odd
            row_indexes = 2+padding:2:size(curr, 1)-padding;
        else
            row_indexes = 1+padding:2:size(curr, 1)-padding;
        end
        
        for row = row_indexes
            for col = 1+padding:size(curr, 2)-padding
                if abs(prev(row, col, :) - post(row, col, :)) > 0.0340
                    prev(row, col, :) = new_prev(row, col, :);
                    post(row, col, :) = new_post(row, col, :);
                    abc_count = abc_count + 1;
                end
            end
        end
        %}
        
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
                
                if mod(frameCnt, 2) == is_odd
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
                
                %{
                tmp = abs(input_patches(2:4:end, :, :, count) - input_patches(4:4:end, :, :, count));
                % mean columns
                tmp = mean(tmp, 2); 
                
                % TODO: row indexes
                row_indexes;
                                
                % prev
                indexes = 2:4:size(input_patches, 1);
                indexes = indexes(tmp > 0.0340);
                %}
            end            
        end
    end
end

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

function [im_h_patches, im_h_dsn_patches] = predict_patches(net, input_patches, label_patches, deinterlaced_patches)
    net.forward({input_patches, label_patches, deinterlaced_patches});
    im_h_patches = net.blobs('output-combine').get_data();
    im_h_dsn_patches = im_h_patches(:, :, :);
end