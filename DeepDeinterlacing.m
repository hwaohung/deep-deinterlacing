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

function [] = init_net(net, h, w, input_channels, eachCnt)
    % Reshape blobs
    net.blobs('input').reshape([(h/2)*4+1 w input_channels eachCnt]);
    net.blobs('label').reshape([h/2 w 1 eachCnt]);
    net.blobs('deinterlace').reshape([h/2 w 1 eachCnt]);
    net.reshape();
end

function [im_hs, im_h_dsns] = patch_deinterlace(frames, input_channels, iter)
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
    net_model = [model_dir 'patch/'];
    if input_channels == 3
        net_model = [net_model 'DeepDeinterlacing_mat31.prototxt'];
    elseif input_channels == 1
        net_model = [net_model 'DeepDeinterlacing_mat11.prototxt'];
    end
    
    %{
    % odd
    net_weights1 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(1).caffemodel'];
    net_weights2 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(1).caffemodel'];
    % even
    net_weights3 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(3).caffemodel'];
    net_weights4 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '(3).caffemodel'];
    %}
    net_weights1 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '.caffemodel'];
    net_weights2 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '.caffemodel'];
    net_weights3 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '.caffemodel'];
    net_weights4 = [model_dir 'snapshots/snapshot_iter_' num2str(iter) '.caffemodel'];

    if ~exist(net_weights1, 'file') || ~exist(net_weights2, 'file') || ~exist(net_weights3, 'file') || ~exist(net_weights4, 'file')
        error('Please check caffemodel is exist or not.');
    end
        
    net_cand1 = caffe.Net(net_model, net_weights1, 'test');
    net_cand2 = caffe.Net(net_model, net_weights2, 'test');
    net_cand3 = caffe.Net(net_model, net_weights3, 'test');
    net_cand4 = caffe.Net(net_model, net_weights4, 'test');

    input_channels = 1;
    [input_patches, label_patches, deinterlaced_patches, eachCnt] = patch2patch(frames, [h w], stride, input_channels);
    
    init_net(net_cand1, h, w, input_channels, eachCnt);
    init_net(net_cand2, h, w, input_channels, eachCnt);
    init_net(net_cand3, h, w, input_channels, eachCnt);
    init_net(net_cand4, h, w, input_channels, eachCnt);   
    
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
        if mod(i, 2)
            net1 = net_cand1;
            net2 = net_cand2;
        else
            net1 = net_cand3;
            net2 = net_cand4;
        end
        
        [im_h_patches1, im_h_dsn_patches1] = predict_patches(net1, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
                                                             deinterlaced_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt));
                                                       
        [im_h_patches2, im_h_dsn_patches2] = predict_patches(net2, input_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), label_patches(:, :, :, (i-1)*eachCnt+1:i*eachCnt), ...
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
                %{
                tmp = abs(input_patches(:, :, 1, (i-1)*eachCnt+j) - input_patches(:, :, 3, (i-1)*eachCnt+j));
                if mean(tmp(:)) <= 0.0318
                    im_hs(row:row+h-1, col:col+w-1, i) = im_h_patches1(:, :, j);
                    im_h_dsns(row:row+h-1, col:col+w-1, i) = im_h_dsn_patches1(:, :, j);
                else
                    im_hs(row:row+h-1, col:col+w-1, i) = im_h_patches2(:, :, j);
                    im_h_dsns(row:row+h-1, col:col+w-1, i) = im_h_dsn_patches2(:, :, j);
                end
                %}
                im_hs(row_indexes, col:col+w-1, i) = im_h_patches1(:, :, j);
                im_h_dsns(row_indexes, col:col+w-1, i) = im_h_dsn_patches1(:, :, j);
            end
        end
    end
end

function [im_h_patches, im_h_dsn_patches] = predict_patches(net, input_patches, label_patches, deinterlaced_patches)
    net.forward({input_patches, label_patches, deinterlaced_patches});
    im_h_patches = net.blobs('output-combine').get_data();
    im_h_dsn_patches = im_h_patches(:, :, :);
end
