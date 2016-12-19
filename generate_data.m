function [] = generate_data()
    clear;
    close all;
    
    %% Settings
    patch_method = 1;
    is_train_data = 1;
    testFramesCnt = 100;
    
    % stride, window(1) must be even(even shift for sure the same parity)
    if patch_method == 1
        window = [16, 16];
        stride = 16;
    else
        window = 3;
    end

    if is_train_data
        folder = 'GenData\Train';
        savepath = 'GenData\train.h5';
        chunksz = 1024;
    else
        folder = 'GenData\Test';
        savepath = 'GenData\test.h5';
        chunksz = 1024;
    end
    
    filepaths = dir(fullfile(folder,'*.avi'));
    
    if patch_method == 1
        gen_patch2patch_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, stride);
    else
        gen_patch2pixel_data(folder, filepaths, savepath, chunksz, testFramesCnt, window);
    end
end

function [] = gen_patch2pixel_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, input_channels)
    for i = 1:length(filepaths)
        frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
           
        [input_patches1, label_patches1] = patch2pixel(frames, window, input_channels);
        
        if i == 1
        	input_data = input_patches1;
        	label_data = label_patches1;
        else
            input_data = cat(4, input_data, input_patches1);
            label_data = cat(4, label_data, label_patches1);
        end
    end
    
    save2hdf5(savepath, chunksz, input_data, label_data);
end

function [] = gen_patch2patch_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, stride)
    for i = 1:length(filepaths)
        frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
        
        [input_patches1, label_patches1, deinterlaced_patches1, eachCnt] = patch2patch(frames, window, stride);
        
        if i == 1
        	input_data = input_patches1;
            label_data = label_patches1;
            deinterlaced_data = deinterlaced_patches1;
        else
            input_data = cat(4, input_data, input_patches1);
            label_data = cat(4, label_data, label_patches1);
            deinterlaced_data = cat(4, deinterlaced_data, deinterlaced_patches1);
        end
    end
    
    %% TODO: Use only related field
    %{
    for i = size(input_data, 4):-1:1
        tmp = abs(input_data(2:4:end, :, :, i) - input_data(4:4:end, :, :, i));
        diffs(i) = mean(tmp(:));
    end
    %}
    
    %{
    disp(mean(diffs))
    indexes = diffs > 0.0338;
    input_data = input_data(:, :, :, indexes(:));
    label_data = label_data(:, :, :, indexes(:));
    deinterlaced_data = deinterlaced_data(:, :, :, indexes(:));
    %}
    
    save2hdf5(savepath, chunksz, input_data, label_data, deinterlaced_data);
end
