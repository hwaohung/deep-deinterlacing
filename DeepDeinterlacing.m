% im_1: gray image
function [im_hs, im_h_dsns, im_fusions, running_time] = DeepDeinterlacing(frames, use_gpu, iter)
    % Set caffe mode
    if exist('use_gpu', 'var') && use_gpu
        caffe.set_mode_gpu();
        % use the first gpu in this demo
        gpu_id = 0;
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu();
    end

    % Initialize the network
    model_dir = 'models/';
    net_model = [model_dir 'DeepDeinterlacing_mat.prototxt'];
    net_weights = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '.caffemodel'];
    phase = 'test'; % run with phase test (so that dropout isn't applied)

    if ~exist(net_weights, 'file')
        error('Please check caffemodel is exist or not.');
    end

    net = caffe.Net(net_model, net_weights, phase);

    %{
    % get label padding
    data_shape = net.blobs('data').shape;
    label_shape = net.blobs('label').shape;
    padding = (data_shape(1) - label_shape(1)) / 2;
    %}
    
    isPatch = 0;  
    
    if isPatch
        % Patch size
        [h ,w] = [100, 100];
        [input_patchs, label_patchs, interlaced_patchs, deinterlaced_patchs, inv_mask_patchs, eachCnt] = prepare_data(frames, [h w], [h w], min([h w]));
    else
        [h w c] = size(frames);
        % TODO: Make sure one patch
        [input_patchs, label_patchs, interlaced_patchs, deinterlaced_patchs, inv_mask_patchs, eachCnt] = prepare_data(frames, [h w], [h w], max([h w]));     
    end

    % Reshape blobs
    net.blobs('input').reshape([h w 3 1]);
    net.blobs('label').reshape([h w 1 1]);
    net.blobs('interlace').reshape([h w 1 1]);
    net.blobs('deinterlace').reshape([h w 1 1]);
    net.blobs('inv-mask').reshape([h w 1 1]);
    net.reshape;
    
    im_hs = zeros(size(frames));
    im_h_dsns = zeros(size(frames));
    tic;
    for i = 1:size(input_patchs, 4)/eachCnt           
        [im_h, im_h_dsn] = deepdeinterlace(net, input_patchs(:, :, :, i:i+eachCnt-1), label_patchs(:, :, :, i:i+eachCnt-1), ...
                                           interlaced_patchs(:, :, :, i:i+eachCnt-1), deinterlaced_patchs, inv_mask_patchs(:, :, :, i:i+eachCnt-1));
            
        im_hs(:, :, i) = reshape(im_h, size(frames, 1), size(frames, 2));
        im_h_dsns(:, :, i) = reshape(im_h_dsn, size(frames, 1), size(frames, 2));           
    end
    running_time = toc;
    
    im_hs = uint8(im_hs * 255);
    im_h_dsns = uint8(im_h_dsns * 255);
    im_fusions = (im_hs * 0.7) + (im_h_dsns * 0.3);

    % call caffe.reset_all() to reset caffe
    caffe.reset_all();
end

function [im_h, im_h_dsn, running_time] = deepdeinterlace(net, input_patchs, label_patchs, interlaced_patchs, deinterlaced_patchs, inv_mask_patchs)
    [h] = size(input_patchs);
    for i = 1:size(input_patchs, 4)
        input_full = input_patchs(:, :, :, i);
        label_full = label_patchs(:, :, :, i);
        interlace_full = interlaced_patchs(:, :, :, i);
        deinterlace_full = deinterlaced_patchs(:, :, :, i);
        inv_mask_full = inv_mask_patchs(:, :, :, i);
            
        % Feed to caffe and get output data
        tic;
        net.forward({input_full, label_full, interlace_full, deinterlace_full, inv_mask_full});
        running_time = toc;
    
        im_h_patch = net.blobs('output-combine').get_data();
        im_h_dsn_patch = net.blobs('output-dsn-combine').get_data();
        im_h((i-1)*h+1:i*h, :) = im_h_patch;
        im_h_dsn((i-1)*h+1:i*h, :) = im_h_dsn_patch;
    end
end

function [input_patchs, label_patchs, interlaced_patchs, deinterlaced_patchs, inv_mask_patchs, eachCnt] = prepare_data(frames, input_size, label_size, stride)
    [hei, wid, cnt] = size(frames);
    %% Get frames, interlaced_fields, inv_masks, deinterlaced_fields
    for frameCnt = 1:cnt
        frame = frames(:, :, frameCnt);
        [interlaced_field, inv_mask] = interlace(frame, mod(frameCnt, 2));
        deinterlaced_field = deinterlace(interlaced_field, mod(frameCnt, 2));
        
        interlaced_fields(:, :, frameCnt) = interlaced_field;
        deinterlaced_fields(:, :, frameCnt) = deinterlaced_field;
        inv_masks(:, :, frameCnt) = inv_mask;
    end
    
    frames = im2double(frames);
    interlaced_fields = im2double(interlaced_fields);        
    deinterlaced_fields = im2double(deinterlaced_fields);
    inv_masks = im2double(inv_masks);
    
    %% Initialization
    input_patchs = zeros(input_size(1), input_size(2), 3, 1);
    label_patchs = zeros(label_size(1), label_size(2), 1, 1);
    interlaced_patchs = zeros(input_size(1), input_size(2), 1, 1);
    deinterlaced_patchs = zeros(input_size(1), input_size(2), 1, 1);
    inv_mask_patchs = zeros(input_size(1), input_size(2), 1, 1);
    count = 0;
    
    %% Generate data pacth
    for frameCnt = 1:cnt
        % Get prev, post field
        if frameCnt == 1
            prev = deinterlaced_fields(:, :, frameCnt);
            post = deinterlaced_fields(:, :, frameCnt+1);
        elseif frameCnt == cnt
            prev = deinterlaced_fields(:, :, frameCnt-1);
            post = deinterlaced_fields(:, :, frameCnt);
        else
            prev = deinterlaced_fields(:, :, frameCnt-1);
            post = deinterlaced_fields(:, :, frameCnt+1);
        end
        
        interlace_full = interlaced_fields(:, :, frameCnt);
        deinterlace_full = deinterlaced_fields(:, :, frameCnt);
        inv_mask_full = inv_masks(:, :, frameCnt);
        
        input_full = reshape([prev, deinterlace_full, post], hei, wid, 3);
        label_full = frames(:, :, frameCnt);
        
        % Test code for check image is ok
        %{
        if frameCnt == 2
            figure(1), imshow(input_full); title('Input Image');
            figure(2), imshow(label_full); title('Label Image');
            figure(3), imshow(interlace_full); title('Interlace Image');
            figure(4), imshow(deinterlace_full); title('De-interlace Image');
            figure(5), imshow(inv_mask_full); title('Mask Image');
            pause;
        end
        %}
                
            %% Generate patchs from each
        % Record each frame has how many patches
        eachCnt = 0;
        for x = 1:stride:hei-input_size+1
            for y = 1:stride:wid-input_size+1
                eachCnt = 1;
                count = count + 1;
                input_patchs(:, :, :, count) = input_full(x:x+input_size(1)-1, y:y+input_size(2)-1, 1:3);
                label_patchs(:, :, :, count) = label_full(x:x+label_size(1)-1, y:y+label_size(2)-1, 1);
                interlaced_patchs(:, :, :, count) = interlace_full(x:x+input_size(1)-1, y:y+input_size(2)-1, 1);
                deinterlaced_patchs(:, :, :, count) = deinterlace_full(x:x+input_size(1)-1, y:y+input_size(2)-1, 1);
                inv_mask_patchs(:, :, :, count) = inv_mask_full(x:x+input_size(1)-1, y:y+input_size(2)-1, 1);
            end
        end
    end
end