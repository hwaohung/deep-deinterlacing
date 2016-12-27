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
        [im_dds, im_fusions] = patch_deinterlace(frames, iter);
    else
        [im_dds, im_fusions] = pixel_deinterlace(net, frames, input_channels);
    end
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
