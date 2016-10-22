% im_1: gray image
function [im_hs, im_h_dsns, im_fusions, running_time] = DeepDeinterlacing(frames, iter)
    use_gpu = 1;
    
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
    
    net_model = [model_dir 'pixel/'];
    net_model = [net_model 'DeepDeinterlacing_mat11.prototxt'];
    
    net_weights = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '.caffemodel'];

    if ~exist(net_weights, 'file')
        error('Please check caffemodel is exist or not.');
    end

    net = caffe.Net(net_model, net_weights, 'test');
    
    tic;
    [im_hs, im_h_dsns] = new_pixel_deinterlace(frames, iter);
    running_time = toc;
    
    im_hs = uint8(im_hs * 255);
    im_h_dsns = uint8(im_h_dsns * 255);
    im_fusions = (im_hs * 0.7) + (im_h_dsns * 0.3);

    % call caffe.reset_all() to reset caffe
    caffe.reset_all();
end

function [im_hs, im_h_dsns] = new_pixel_deinterlace(frames, iter)
    % Initialize the network
    model_dir = 'models/';
    
    net_model = [model_dir 'pixel/' 'DeepDeinterlacing_mat11.prototxt'];
    
    net_weights1 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(1).caffemodel'];
    net_weights2 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(2).caffemodel'];

    if ~exist(net_weights1, 'file') || ~exist(net_weights2, 'file')
        error('Please check caffemodel is exist or not.');
    end

    net1 = caffe.Net(net_model, net_weights1, 'test');
    net2 = caffe.Net(net_model, net_weights2, 'test');

    cutting = 8;
    
    [h, w, c] = size(frames);
    tic;
    [input_patches, label_patches, deinterlaced_patches, eachCnt] = new_patch2pixel(frames); 
    run_time = toc;
    disp(run_time);
    
    total = (h/2)*w;
    step = int32(total/cutting);
    
    % Reshape blobs
    net1.blobs('input').reshape([4 4 1 step]);
    net1.blobs('label').reshape([1 1 1 step]);
    net1.reshape();
    
    net2.blobs('input').reshape([4 4 1 step]);
    net2.blobs('label').reshape([1 1 1 step]);
    net2.reshape();
    
    frames = im2double(frames);
    im_hs = zeros(size(frames));
    im_h_dsns = zeros(size(frames));
    for i = 1:c
        curr_input_patches = input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt);
        curr_label_patches = label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt);        
        mask = deinterlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt);
        indexes = eval_t(curr_input_patches, 0.027, true);
        
        for j = 1:step:total 
            [tmp1] = predict_patches(net1, curr_input_patches(:, :, :, j:j+step-1), curr_label_patches(:, :, :, j:j+step-1));
            [tmp2] = predict_patches(net2, curr_input_patches(:, :, :, j:j+step-1), curr_label_patches(:, :, :, j:j+step-1));
            tmp = tmp1 .* (1-indexes(:, :, :, j:j+step-1)) + tmp2 .* (indexes(:, :, :, j:j+step-1));
            
            if j == 1
                im_h_patches = tmp;
            else
                im_h_patches = cat(4, im_h_patches, tmp);
            end
        end
        
        im_h_patches = reshape(im_h_patches, w, h/2);
        im_h_patches = transpose(im_h_patches);
        mask = reshape(mask, w, h/2);
        mask = transpose(mask);
        inv_mask = 1 - mask;
        
        % TODO: Use processed data for accerlate
        im_hs(:, :, i) = im2double(deinterlace(frames(:, :, i), mod(i, 2)));
                
        if mod(i, 2) == 1
            im_hs(2:2:end, :, i) = im_hs(2:2:end, :, i) .* mask + im_h_patches .* inv_mask;
        else
            im_hs(1:2:end, :, i) = im_hs(1:2:end, :, i) .* mask + im_h_patches .* inv_mask;
        end
        
        im_h_dsns = im_hs;
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
            im_h_dsn_patches(:, :, i) = im_h_patches(:, :, i);
            %im_h_dsn_patches(:, :, i) = net.blobs('output-dsn-combine').get_data();
        end
    else
        net.forward({input_patches, label_patches});
        im_h_patches = net.blobs('output-combine').get_data();
        im_h_dsn_patches = im_h_patches;
        %im_h_dsn_patches = net.blobs('output-dsn-combine').get_data();
    end
end
