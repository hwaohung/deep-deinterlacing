% im_1: gray image
function [im_hs, im_h_dsns, im_fusions, running_time] = DeepDeinterlacing(frames, iter)
    use_gpu = 1;
    patch_method = 1;
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

    % Initialize the network
    model_dir = 'models/';
    
    if patch_method == 1
        net_model = [model_dir 'patch/'];
    else
        net_model = [model_dir 'pixel/'];
    end
    
    if input_channels == 3
        net_model = [net_model 'DeepDeinterlacing_mat31.prototxt'];
    elseif input_channels == 1
        net_model = [net_model 'DeepDeinterlacing_mat11.prototxt'];
    end
    
    net_weights = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '.caffemodel'];

    if ~exist(net_weights, 'file')
        error('Please check caffemodel is exist or not.');
    end

    net = caffe.Net(net_model, net_weights, 'test');

    tic; 
    if patch_method == 1
        [im_hs, im_h_dsns] = patch_deinterlace(frames, input_channels, iter);
    else
        [im_hs, im_h_dsns] = pixel_deinterlace(net, frames, input_channels);
    end
    running_time = toc;
    
    im_hs = im2uint8(im_hs);
    im_h_dsns = im2uint8(im_h_dsns);
    im_fusions = (im_hs * 0.7) + (im_h_dsns * 0.3);

    % Reset caffe
    caffe.reset_all();
end

function [im_hs, im_h_dsns] = patch_deinterlace(frames, input_channels, iter)
    isPatch = 0;

    if isPatch
        % Patch size
        h = 30;
        w = 30;
        stride = min([h w]);
    else
        [h w c] = size(frames);
        stride = max([h w]);
    end
    
    [input_patches, label_patches, interlaced_patches, deinterlaced_patches, inv_mask_patches, eachCnt] = patch2patch(frames, [h w], input_channels); 
    
    model_dir = 'models/';
    net_model = [model_dir 'patch/'];
    if input_channels == 3
        net_model = [net_model 'DeepDeinterlacing_mat31.prototxt'];
    elseif input_channels == 1
        net_model = [net_model 'DeepDeinterlacing_mat11.prototxt'];
    end
    
    net_weights1 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(1).caffemodel'];
    net_weights2 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '.caffemodel'];

    if ~exist(net_weights1, 'file') || ~exist(net_weights2, 'file')
        error('Please check caffemodel is exist or not.');
    end

    net1 = caffe.Net(net_model, net_weights1, 'test');
    net2 = caffe.Net(net_model, net_weights2, 'test');
    
    % Reshape blobs
    net1.blobs('input').reshape([h w input_channels eachCnt]);
    net1.blobs('label').reshape([h w 1 eachCnt]);
    net1.blobs('interlace').reshape([h w 1 eachCnt]);
    net1.blobs('deinterlace').reshape([h w 1 eachCnt]);
    net1.blobs('inv-mask').reshape([h w 1 eachCnt]);
    net1.reshape();
    
    net2.blobs('input').reshape([h w input_channels eachCnt]);
    net2.blobs('label').reshape([h w 1 eachCnt]);
    net2.blobs('interlace').reshape([h w 1 eachCnt]);
    net2.blobs('deinterlace').reshape([h w 1 eachCnt]);
    net2.blobs('inv-mask').reshape([h w 1 eachCnt]);
    net2.reshape();
    
    im_hs = zeros(size(frames));
    im_h_dsns = zeros(size(frames));
    for i = 1:size(input_patches, 4)/eachCnt
        [im_h_patches1, im_h_dsn_patches1] = predict_patches(net1, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                                             interlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), deinterlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                                             inv_mask_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt));
                                                       
        [im_h_patches2, im_h_dsn_patches2] = predict_patches(net2, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                                             interlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), deinterlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                                             inv_mask_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt));
        
        s_row = 1;
        s_col = 1;
        next = false;
        for j = 1:eachCnt
            if s_col > size(frames, 2)
                s_row = s_row + stride;
                s_col = 1;
            elseif s_col + w - 1 > size(frames, 2)
                s_col = size(frames, 2) - w + 1;
                next = true;
            end

            if s_row + h - 1 > size(frames, 1)
                s_row = size(frames, 1) - h + 1;
            end
            
            % TODO
            tmp = abs(input_patches(:, :, 1, (i-1)*eachCnt+j) - input_patches(:, :, 3, (i-1)*eachCnt+j));
            if sum(tmp(:)) <= 28.4727
                im_hs(s_row:s_row+h-1, s_col:s_col+w-1, i) = im_h_patches1(:, :, j);
                im_h_dsns(s_row:s_row+h-1, s_col:s_col+w-1, i) = im_h_dsn_patches1(:, :, j);
            else
                im_hs(s_row:s_row+h-1, s_col:s_col+w-1, i) = im_h_patches2(:, :, j);
                im_h_dsns(s_row:s_row+h-1, s_col:s_col+w-1, i) = im_h_dsn_patches2(:, :, j);
            end
                                    
            if next
                s_row = s_row + stride;
                s_col = 1;
                next = false;
            else
                s_col = s_col + stride;
            end
        end
    end
end

function [im_h_patches, im_h_dsn_patches] = predict_patches(net, input_patches, label_patches, interlaced_patches, deinterlaced_patches, inv_mask_patches)
    net.forward({input_patches, label_patches, interlaced_patches, deinterlaced_patches, inv_mask_patches});
    im_h_patches = net.blobs('output-combine').get_data();
    im_h_dsn_patches = im_h_patches(:, :, :);
end
