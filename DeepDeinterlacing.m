% im_1: gray image
function [im_h, im_h_dsn, im_fusion, running_time] = DeepDeinterlacing(im_l, odd, use_gpu, iter)
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
net_weights = [model_dir 'DeepDeinterlacing/snapshot_iter_' num2str(iter) '.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)

if ~exist(net_weights, 'file')
    error('Please check caffemodel is exist or not.');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

%{
% get label padding
data_shape = net.blobs('data').shape;
label_shape = net.blobs('label').shape;
padding = (data_shape(1) - label_shape(1)) / 2;
%}

%{
% prepare data and label
if  strcmp(class(im_l), 'uint8')
    im_l = im2double(im_l);
    im_l = single(im_l);
end
%}

[im_input, im_mask] = interlace(im_l, odd);
im_input_i = deinterlace(im_input, odd);
im_label = im_l;

im_input = im2double(im_input);
im_input_i = im2double(im_input_i);
im_mask = im2double(im_mask);
im_label = im2double(im_l);


[w h c] = size(im_input);
isPatch = 1;
patch_size = 100;
stride = 80;
% TODO: 1 channel
input_channel = 3;

if isPatch
    % reshape blobs
    net.blobs('data').reshape([patch_size patch_size input_channel 1]);
    net.blobs('data-i').reshape([patch_size patch_size input_channel 1]);
    net.blobs('inv-mask').reshape([patch_size patch_size input_channel 1]);
    net.blobs('label').reshape([patch_size patch_size input_channel 1]);
    net.reshape;
    
    for iw = 1:stride:w
        for ih=1:stride:h
            rang_w_start = iw;
            rang_h_start = ih;
            if iw+patch_size >= w
                rang_w_start = w-patch_size + 1;
            end
            
            if(ih+patch_size >= h)
                rang_h_start = h-patch_size + 1;
            end
            
            im_input_patch = im_input(rang_w_start:rang_w_start + patch_size-1, rang_h_start:rang_h_start + patch_size-1, 1:input_channel);
            im_input_i_patch = im_input_i(rang_w_start:rang_w_start + patch_size-1, rang_h_start:rang_h_start + patch_size-1, 1:input_channel);
            im_mask_patch = im_mask(rang_w_start:rang_w_start + patch_size-1, rang_h_start:rang_h_start + patch_size-1, 1:input_channel);
            im_label_patch = im_label(rang_w_start:rang_w_start + patch_size-1, rang_h_start:rang_h_start + patch_size-1, 1:input_channel);

            % run deinterlacing
            tic;
            % note: interlace, naive deinterlace, mask, label
            net.forward({im_input_patch, im_input_i_patch, im_mask_patch, im_label_patch});
            running_time = toc;
            
            % get output data
            im_h_patch = net.blobs('output-combine').get_data();
            im_h(rang_w_start:rang_w_start+patch_size-1, rang_h_start:rang_h_start+patch_size-1, 1:input_channel) = im_h_patch;
            im_h_dsn_patch = net.blobs('output-dsn-combine').get_data();
            im_h_dsn(rang_w_start:rang_w_start+patch_size-1, rang_h_start:rang_h_start+patch_size-1, 1:input_channel) = im_h_dsn_patch;
        end
    end
else    
    %%%%%%% Maybe has some error%%%%%%%%
    % reshape blobs
    net.blobs('data').reshape([w h input_channel 1]);
    net.blobs('data-i').reshape([w h input_channel 1]);
    net.blobs('inv-mask').reshape([w h input_channel 1]);
    net.blobs('label').reshape([w h input_channel 1]);
    net.reshape;

    % run Deinterlacing
    tic;
    net.forward({im_input, im_input_i, im_mask, im_label});
    running_time = toc;
    
    %get output data
    im_h = net.blobs('output-combine').get_data();
    im_h_dsn = net.blobs('output-dsn-combine').get_data();
end

im_h = uint8(im_h * 255);
im_h_dsn = uint8(im_h_dsn * 255);
im_fusion = (im_h * 0.7) + (im_h_dsn * 0.3);

% call caffe.reset_all() to reset caffe
caffe.reset_all();
end