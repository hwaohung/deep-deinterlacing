function [im_dds, im_fusions, running_time] = DeepDeinterlacing(frames, iter)
    patch_method = 1;
    
    % Set caffe gpu mode
    caffe.set_mode_gpu();
    % use the first gpu in this demo
    gpu_id = 0;
    caffe.set_device(gpu_id);
    
    % Set caffe cpu mode
    %caffe.set_mode_cpu();
    
    tic; 
    if patch_method == 1
        %[im_dds, im_fusions] = patch_deinterlace(frames, iter);
        [im_dds, im_fusions] = patch_deinterlace_MD(frames, iter);
    else
        [im_dds, im_fusions] = pixel_deinterlace(net, frames, input_channels);
    end
    
    [im_dds] = MNN(im_dds, im_fusions, frames);
    
    running_time = toc;
    
    im_dds = im2uint8(im_dds);
    im_fusions = im2uint8(im_fusions);

    % Reset caffe
    caffe.reset_all();
end

function [] = init_net(net, h, w, input_channels, eachCnt)
    % Reshape blobs
    net.blobs('input').reshape([(h/2)*4+1 w input_channels eachCnt]);
    net.blobs('label').reshape([h/2 w input_channels eachCnt]);
    net.blobs('deinterlace').reshape([h/2 w input_channels eachCnt]);
    net.reshape();
end

function [im_dds, im_fusions] = patch_deinterlace_MD(frames, iter)
    isPatch = 0;

    % stride, window(1) must be even(even shift for sure the same parity)
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
    net_model = [model_dir 'patch/'];
    net_model = [net_model 'DeepDeinterlacing_mat.prototxt'];
    
    net_weights1 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(1).caffemodel'];
    net_weights2 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(2).caffemodel'];

    if ~exist(net_weights1, 'file') || ~exist(net_weights2, 'file')
        net_weights1
        error('Please check caffemodel is exist or not.');
    end
        
    net1 = caffe.Net(net_model, net_weights1, 'test');
    net2 = caffe.Net(net_model, net_weights2, 'test');
    
    [input_patches, label_patches, deinterlaced_patches, flags, eachCnt] = genTemp(frames, [h w], stride, true);
    %[input_patches, label_patches, deinterlaced_patches, flags, eachCnt] = patch2patch(frames, [h w], stride);
    
    init_net(net1, h, w, 3, eachCnt);
    init_net(net2, h, w, 3, eachCnt);
    
    [hei, wid, ch, cnt] = size(frames);
    rows = (1:stride:hei-h+1);
    if rows(end) ~= hei-h+1
        rows(end+1) = hei-h+1;
    end
    
    cols = (1:stride:wid-w+1);
    if cols(end) ~= wid-w+1
        cols(end+1) = wid-w+1;
    end
    
    im_dds = im2double(frames);
    im_fusions = im2double(frames);
    
    for i = 1:size(input_patches, 4)/eachCnt
        [im_dd_patches] = predict_patches(net1, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                           label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                           deinterlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt));
                                                                         
        j = 0;
        for row = rows
            % odd => fill even
            if mod(i, 2)
                row_indexes = (row+1:2:row+h-1);
            % even => fill odd
            else
                row_indexes = (row:2:row+h-1);
            end
            
            for col = cols
                j = j + 1;
                col_indexes = (col:col+w-1);
                
                im_dds(row_indexes, col_indexes, :, i) = im_dd_patches(:, :, :, j);
            end
        end
    end
    
    [input_patches, label_patches, deinterlaced_patches, flags, eachCnt] = genTemp(frames, [h w], stride, false);
    for i = 1:size(input_patches, 4)/eachCnt
        [im_dd_patches] = predict_patches(net2, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                           label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                           deinterlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt));
                                                                         
        j = 0;
        for row = rows
            % odd => fill even
            if mod(i, 2)
                row_indexes = (row+1:2:row+h-1);
            % even => fill odd
            else
                row_indexes = (row:2:row+h-1);
            end
            
            for col = cols
                j = j + 1;
                col_indexes = (col:col+w-1);
                
                im_fusions(row_indexes, col_indexes, :, i) = im_dd_patches(:, :, :, j);       
            end
        end
    end
end

function [im_dds, im_fusions] = patch_deinterlace(frames, iter)
    isPatch = 0;

    % stride, window(1) must be even(even shift for sure the same parity)
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
    net_model = [model_dir 'patch/'];
    net_model = [net_model 'DeepDeinterlacing_mat.prototxt'];
    
    net_weights1 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(1).caffemodel'];
    net_weights2 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(2).caffemodel'];

    if ~exist(net_weights1, 'file') || ~exist(net_weights2, 'file')
        error('Please check caffemodel is exist or not.');
    end
        
    net1 = caffe.Net(net_model, net_weights1, 'test');
    net2 = caffe.Net(net_model, net_weights2, 'test');

    [input_patches, label_patches, deinterlaced_patches, flags, eachCnt] = patch2patch(frames, [h w], stride);
    
    init_net(net1, h, w, 3, eachCnt);
    init_net(net2, h, w, 3, eachCnt);
    
    [hei, wid, ch, cnt] = size(frames);
    rows = (1:stride:hei-h+1);
    if rows(end) ~= hei-h+1
        rows(end+1) = hei-h+1;
    end
    
    cols = (1:stride:wid-w+1);
    if cols(end) ~= wid-w+1
        cols(end+1) = wid-w+1;
    end
    
    im_dds = im2double(frames);
    im_fusions = im2double(frames);
    for i = 1:size(input_patches, 4)/eachCnt
        flag = flags(:, :, :, (i-1)*eachCnt+1:i*eachCnt);
        [im_dd_patches1] = predict_patches(net1, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                           label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                           deinterlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt));
                                       
        [im_dd_patches2] = predict_patches(net2, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                           label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                           deinterlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt));
                                       
        j = 0;
        for row = rows
            % odd => fill even
            if mod(i, 2)
                row_indexes = (row+1:2:row+h-1);
            % even => fill odd
            else
                row_indexes = (row:2:row+h-1);
            end
            
            for col = cols
                j = j + 1;
                col_indexes = (col:col+w-1);
                
                if flag(:, :, :, j) < Var.T
                    im_dds(row_indexes, col_indexes, :, i) = im_dd_patches1(:, :, :, j);
                    im_fusions(row_indexes, col_indexes, :, i) = im_dd_patches1(:, :, :, j);
                else
                    im_dds(row_indexes, col_indexes, :, i) = im_dd_patches2(:, :, :, j);
                    im_fusions(row_indexes, col_indexes, :, i) = im_dd_patches2(:, :, :, j);
                end
                
            end
        end
    end
end

function [im_dd_patches] = predict_patches(net, input_patches, label_patches, deinterlaced_patches)
    net.forward({input_patches, label_patches, deinterlaced_patches});
    im_dd_patches = net.blobs('output-combine').get_data();
end

function [im_fusions] = MNN(im_nn1s, im_nn2s, frames)
    [hei, wid, ch, cnt] = size(im_nn1s);

    [deinterlaced_fields, prpo_fields] = deinterlace_video(im2uint8(im_nn1s), cnt);
    deinterlaced_fields = im2double(deinterlaced_fields);
    prpo_fields = im2double(prpo_fields);
    frames = im2double(frames);
    
    maMask = [0.1, 0.2, 0.4, 0.2, 0.1];
    maMask = ones(7, 7);
    maMask(1:2:end, :) = 0;
    maMask = maMask / sum(maMask(:));
    mnnMask = [0.2, 0.2, 0.2, 0.2, 0.2];
    
    ccc = 0;
    diffs1 = zeros(1);
    
    % TODO: Check type 'double'
    im_fusions = im_nn1s(:, :, :, :);
    for fCnt = 1:cnt
        if fCnt == 1
            prev2 = deinterlaced_fields(:, :, :, fCnt+2);
            prev1 = deinterlaced_fields(:, :, :, fCnt+1);
            post1 = deinterlaced_fields(:, :, :, fCnt+1);
            post2 = deinterlaced_fields(:, :, :, fCnt+2);
        elseif fCnt == 2
            prev2 = deinterlaced_fields(:, :, :, fCnt+2);
            prev1 = deinterlaced_fields(:, :, :, fCnt-1);
            post1 = deinterlaced_fields(:, :, :, fCnt+1);
            post2 = deinterlaced_fields(:, :, :, fCnt+2);
        elseif fCnt == cnt-1
            prev2 = deinterlaced_fields(:, :, :, fCnt-2);
            prev1 = deinterlaced_fields(:, :, :, fCnt-1);
            post1 = deinterlaced_fields(:, :, :, fCnt+1);
            post2 = deinterlaced_fields(:, :, :, fCnt-2);
        elseif fCnt == cnt
            prev2 = deinterlaced_fields(:, :, :, fCnt-2);
            prev1 = deinterlaced_fields(:, :, :, fCnt-1);
            post1 = deinterlaced_fields(:, :, :, fCnt-1);
            post2 = deinterlaced_fields(:, :, :, fCnt-2);
        else
            prev2 = deinterlaced_fields(:, :, :, fCnt-2);
            prev1 = deinterlaced_fields(:, :, :, fCnt-1);
            post1 = deinterlaced_fields(:, :, :, fCnt+1);
            post2 = deinterlaced_fields(:, :, :, fCnt+2);
        end
        
        if mod(fCnt, 2)
            sy = 2;
        else
            sy = 1;
        end       
                    
        prpo_field = rgb2gray(prpo_fields(:, :, :, fCnt));
        pr = zeros(hei, wid, class(prpo_field));
        po = zeros(hei, wid, class(prpo_field));
        pr(sy:2:end, :) = prpo_field(1:2:end, :);
        po(sy:2:end, :) = prpo_field(2:2:end, :);
        absdiff = abs(pr - po);
        mam = conv2(absdiff, maMask, 'same') * 255;
        
        t1 = rgb2gray(prev1);
        t2 = rgb2gray(post1);
        tmp = abs(t1 - t2);
        absdiff(sy:2:end, :) = tmp(sy:2:end, :);
        mnnm = conv2(absdiff, mnnMask, 'same') * 255;
        
        for y = sy:2:hei
            for x = 1:wid
                % MA
                ccc = ccc + 1;
                diffs1(ccc) = mam(y, x);
                diffs2(ccc) = mnnm(y, x);
                if mam(y, x) >= Var.T_MA
                    %im_fusions(y, x, :, fCnt) = frames(y, x, :, fCnt);
                    %im_fusions(y, x, :, fCnt) = deinterlaced_fields(y, x, :, fCnt);
                % MNN
                else
                    % TODO: Temp Pass
                    if mnnm(y, x) >= Var.T 
                        %im_fusions(y, x, :, fCnt) = frames(y, x, :, fCnt);
                        im_fusions(y, x, :, fCnt) = im_nn2s(y, x, :, fCnt);
                    end
                end
            end
        end
    end
end
