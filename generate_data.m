function [] = generate_data()
    clear;
    close all;
    
    %% Settings
    is_train_data = 0;
    input_size = [3, 3];
    input_channels = 3;
    testFramesCnt = 30;

    if is_train_data
        folder = 'Train';
        savepath = 'train.h5';
        chunksz = 64;
    else
        folder = 'Test';
        savepath = 'test.h5';
        chunksz = 64;
    end
    
    filepaths = dir(fullfile(folder,'*.avi'));

    %% Generate data
    for i = 1:length(filepaths)
        frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
        [resizeds1, resizeds2] = interlaced_resize(frames);
        
        [input_patches1, label_patches1, eachCnt1] = prepare_data(resizeds1, input_size(1), input_channels);
        [input_patches2, label_patches2, eachCnt2] = prepare_data(resizeds2, input_size(1), input_channels);
        
        if i == 1
            input_data = cat(4, input_patches1, input_patches2);
            label_data = cat(4, label_patches1, label_patches2);
        else
            input_data = cat(4, input_data, input_patches1, input_patches2);
            label_data = cat(4, input_data, label_patches1, label_patches2);
        end
    end
  
    save2hdf5(savepath, chunksz, input_data, label_data);
end

%% writing to HDF5
function [] = save2hdf5(savepath, chunksz, input_data, label_data)
    %% Data order rearrange
    count = size(input_data, 4);
    order = randperm(count);
    input_data = input_data(:, :, :, order); 
    label_data = label_data(:, :, :, order); 

    created_flag = false;
    totalct = 0;

    for batchno = 1:floor(count/chunksz)
        last_read = (batchno-1) * chunksz;

        b_input_data = input_data(:, :, :, last_read+1:last_read+chunksz);
        b_label_data = label_data(:, :, :, last_read+1:last_read+chunksz);
    
        startloc = struct('input_data', [1, 1, 1, totalct+1], ...
                          'label_data', [1, 1, 1, totalct+1]);
        curr_dat_sz = store2hdf5(savepath, chunksz, ~created_flag, startloc, b_input_data, b_label_data); 
        
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    
    h5disp(savepath);
end

%{
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
                          'interlaced_data', [1,1,1,totalct+1], ...
                          'deinterlaced_data', [1,1,1,totalct+1], ...
                          'inv_mask_data', [1,1,1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath, b_input_data, b_label_data, b_interlaced_data, b_deinterlaced_data, b_inv_mask_data, ~created_flag, startloc, chunksz); 
    
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    
    h5disp(savepath);
end
%}
