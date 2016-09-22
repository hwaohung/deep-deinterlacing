function [] = generate_data()
    clear;close all;

    %% Settings
    is_train_data = 1;
    size_input = 31;
    size_label = 31;
    testFramesCnt = 30;

    if is_train_data
        folder = 'Train';
        savepath = 'train.h5';
        stride = 45;
        chunksz = 64;
        filepaths = dir(fullfile(folder,'*.avi'));
    else
        folder = 'Test';
        savepath = 'test.h5';
        stride = 45;
        chunksz = 2;
        filepaths = dir(fullfile(folder,'*.avi'));
    end

    %% Generate data
    for i = 1:length(filepaths)
        frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
        [resizeds1, resizeds2] = resize_frames(frames);        
        [input_patchs, label_patchs, interlaced_patchs, deinterlaced_patchs, inv_mask_patchs] = prepare_data(resizeds1, size_input, size_label, stride);
                
        if i == 1
            input_data = input_patchs;
            label_data = label_patchs;
            interlaced_data = interlaced_patchs;
            deinterlaced_data = deinterlaced_patchs;
            inv_mask_data = inv_mask_patchs;
        else
            input_data = cat(4, input_data, input_patchs);
            label_data = cat(4, label_data, label_patchs);
            interlaced_data = cat(4, interlaced_data, interlaced_patchs);
            deinterlaced_data = cat(4, deinterlaced_data, deinterlaced_patchs);
            inv_mask_data = cat(4, inv_mask_data, inv_mask_patchs);
        end
        
        [input_patchs, label_patchs, interlaced_patchs, deinterlaced_patchs, inv_mask_patchs] = prepare_data(resizeds2, size_input, size_label, stride);
        
        input_data = cat(4, input_data, input_patchs);
        label_data = cat(4, label_data, label_patchs);
        interlaced_data = cat(4, interlaced_data, interlaced_patchs);
        deinterlaced_data = cat(4, deinterlaced_data, deinterlaced_patchs);
        inv_mask_data = cat(4, inv_mask_data, inv_mask_patchs);
    end
  
    save2hdf5(savepath, chunksz, input_data, label_data, interlaced_data, deinterlaced_data, inv_mask_data);
end

% Gray only
function [frames] = get_video_frames(filename, requiredCnt)
    v = VideoReader(filename);    
    if ~exist('requiredCnt', 'var')
        % TODO: Not sure
        requiredCnt = v.FrameRate;
    end
    
    for frameCnt = 1:requiredCnt
    	if ~hasFrame(v)
        	break;
        end
        
        frame = readFrame(v);      
        if size(frame, 3) > 1
            frame = rgb2gray(frame);
        end
        
        % Image even * even size(why)
        frame = modcrop(frame, 2);
        frames(:, :, frameCnt) = frame;
    end
end

function [input_patchs, label_patchs, interlaced_patchs, deinterlaced_patchs, inv_mask_patchs] = prepare_data(frames, size_input, size_label, stride)
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
    input_patchs = zeros(size_input, size_input, 3, 1);
    label_patchs = zeros(size_label, size_label, 1, 1);
    interlaced_patchs = zeros(size_input, size_input, 1, 1);
    deinterlaced_patchs = zeros(size_input, size_input, 1, 1);
    inv_mask_patchs = zeros(size_input, size_input, 1, 1);
    count = 0;
    
    %% Generate data
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
        for x = 1:stride:hei-size_input+1
            for y = 1:stride:wid-size_input+1
                count = count + 1;
                input_patchs(:, :, :, count) = input_full(x:x+size_input-1, y:y+size_input-1, 1:3);
                label_patchs(:, :, :, count) = label_full(x:x+size_label-1, y:y+size_label-1, 1);
                interlaced_patchs(:, :, :, count) = interlace_full(x:x+size_input-1, y:y+size_input-1, 1);
                deinterlaced_patchs(:, :, :, count) = deinterlace_full(x:x+size_input-1, y:y+size_input-1, 1);
                inv_mask_patchs(:, :, :, count) = inv_mask_full(x:x+size_input-1, y:y+size_input-1, 1);
            end
        end
    end
end

%% writing to HDF5
function [] = save2hdf5(savepath, chunksz, input_data, label_data, interlaced_data, deinterlaced_data, inv_mask_data)
    %% Data order rearrange
    count = size(input_data, 4);
    order = randperm(count);
    input_data = input_data(:, :, :, order); 
    label_data = label_data(:, :, :, order); 
    interlaced_data = interlaced_data(:, :, :, order);
    deinterlaced_data = deinterlaced_data(:, :, :, order);
    inv_mask_data = inv_mask_data(:, :, :, order);

    created_flag = false;
    totalct = 0;

    for batchno = 1:floor(count/chunksz)
        last_read = (batchno-1) * chunksz;

        b_input_data = input_data(:, :, :, last_read+1:last_read+chunksz);
        b_label_data = label_data(:, :, :, last_read+1:last_read+chunksz);
        b_interlaced_data = interlaced_data(:, :, :, last_read+1:last_read+chunksz); 
        b_deinterlaced_data = deinterlaced_data(:, :, :, last_read+1:last_read+chunksz); 
        b_inv_mask_data = inv_mask_data(:, :, :, last_read+1:last_read+chunksz); 
    
        startloc = struct('input_data', [1,1,1,totalct+1], ...
                          'label_data', [1,1,1,totalct+1], ...
                          'interlaced_data', [1,1,1,totalct+1],...
                          'deinterlaced_data', [1,1,1,totalct+1],...
                          'inv_mask_data', [1,1,1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath, b_input_data, b_label_data, b_interlaced_data, b_deinterlaced_data, b_inv_mask_data, ~created_flag, startloc, chunksz); 
    
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    
    h5disp(savepath);
end
