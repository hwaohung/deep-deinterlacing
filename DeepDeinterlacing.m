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

    %{
    % get label padding
    data_shape = net.blobs('data').shape;
    label_shape = net.blobs('label').shape;
    padding = (data_shape(1) - label_shape(1)) / 2;
    %}
    
    tic;
    
    if patch_method == 1
        [im_hs, im_h_dsns] = patch_deinterlace(net, frames, input_channels);
    else
        [im_hs, im_h_dsns] = pixel_deinterlace(net, frames, input_channels);
    end
    running_time = toc;
    
    im_hs = uint8(im_hs * 255);
    im_h_dsns = uint8(im_h_dsns * 255);
    im_fusions = (im_hs * 0.7) + (im_h_dsns * 0.3);

    % call caffe.reset_all() to reset caffe
    caffe.reset_all();
end

function [im_hs, im_h_dsns] = pixel_deinterlace(net, frames, input_channels)
    window = 3;
    
    [input_patches, label_patches, eachCnt] = patch2pixel(frames, window, input_channels); 
    
    % Reshape blobs
    net.blobs('input').reshape([window window input_channels 1]);
    net.blobs('label').reshape([1 1 1 1]);
    net.reshape();
    
    im_hs = zeros(size(frames));
    im_h_dsns = zeros(size(frames));
    for i = 1:size(frames, 3)
        [im_h_patches, im_h_dsn_patches] = predict_patches(net, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt));
                       
        im_h_patches = reshape(im_h_patches, size(frames, 2), size(frames, 1)/2);
        im_h_dsn_patches = reshape(im_h_dsn_patches, size(frames, 2), size(frames, 1)/2);
        im_h_patches = transpose(im_h_patches);
        im_h_dsn_patches = transpose(im_h_dsn_patches);
        
        im_hs(:, :, i) = frames(:, :, i);
        im_h_dsns(:, :, i) = frames(:, :, i);
        if mod(i, 2) == 1
            im_hs(2:2:end, :, i) = im_h_patches;
            im_h_dsns(2:2:end, :, i) = im_h_dsn_patches;
        else
            im_hs(1:2:end, :, i) = im_h_patches;
            im_h_dsns(1:2:end, :, i) = im_h_dsn_patches;
        end
    end
end

function [im_hs, im_h_dsns] = patch_deinterlace(net, frames, input_channels)
    isPatch = 1;

    if isPatch
        % Patch size
        h = 32;
        w = 32;
        stride = min([h w]);
    else
        [h w c] = size(frames);
        stride = max([h w]);
    end
    
    [input_patches, label_patches, interlaced_patches, deinterlaced_patches, inv_mask_patches, eachCnt] = patch2patch(frames, [h w], input_channels); 
    
    % Reshape blobs
    net.blobs('input').reshape([h w input_channels 1]);
    net.blobs('label').reshape([h w 1 1]);
    net.blobs('interlace').reshape([h w 1 1]);
    net.blobs('deinterlace').reshape([h w 1 1]);
    net.blobs('inv-mask').reshape([h w 1 1]);
    net.reshape();
    
    im_hs = zeros(size(frames));
    im_h_dsns = zeros(size(frames));
    for i = 1:size(input_patches, 4)/eachCnt
        [im_h_patches, im_h_dsn_patches] = predict_patches(net, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
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
            
            im_hs(s_row:s_row+h-1, s_col:s_col+w-1, i) = im_h_patches(:, :, j);
            im_h_dsns(s_row:s_row+h-1, s_col:s_col+w-1, i) = im_h_dsn_patches(:, :, j);
                        
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
    if exist('interlaced_patches','var')
        for i = 1:size(input_patches, 4)
            input_full = input_patches(:, :, :, i);
            label_full = label_patches(:, :, :, i);
            interlace_full = interlaced_patches(:, :, :, i);
            deinterlace_full = deinterlaced_patches(:, :, :, i);
            inv_mask_full = inv_mask_patches(:, :, :, i);
            
            % Feed to caffe and get output data
            net.forward({input_full, label_full, interlace_full, deinterlace_full, inv_mask_full});
    
            im_h_patches(:, :, i) = net.blobs('output-combine').get_data();
            im_h_dsn_patches(:, :, i) = net.blobs('output-dsn-combine').get_data();
        end
    else
        for i = 1:size(input_patches, 4)
            input_full = input_patches(:, :, :, i);
            label_full = label_patches(:, :, :, i);
            
            % Feed to caffe and get output data
            net.forward({input_full, label_full});
    
            im_h_patches(:, :, i) = net.blobs('output-combine').get_data();
            im_h_dsn_patches(:, :, i) = net.blobs('output-dsn-combine').get_data();
        end
    end
end
