function [] = generate_data()
    clear;
    close all;
    
    %% Settings
    patch_method = 1;
    is_train_data = 0;
    input_channels = 3;
    testFramesCnt = 100;
    
    if patch_method == 1
        window = [32, 32];
        stride = 30;
    else
        window = 3;
    end

    if is_train_data
        folder = 'Train';
        savepath = 'train.h5';
        chunksz = 1024;
    else
        folder = 'Test';
        savepath = 'test.h5';
        chunksz = 1024;
    end
    
    filepaths = dir(fullfile(folder,'*.avi'));
    
    if patch_method == 1
        gen_patch2patch_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, stride, input_channels);
    else
        gen_patch2pixel_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, input_channels);
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

function [] = gen_patch2patch_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, stride, input_channels)
    for i = 1:length(filepaths)
        frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
        
        [input_patches1, label_patches1, interlaced_patches1, deinterlaced_patches1, inv_mask_patches1] = patch2patch(frames, window, stride, input_channels);
        
        if i == 1
        	input_data = input_patches1;
            label_data = label_patches1;
            interlaced_data = interlaced_patches1;
            deinterlaced_data = deinterlaced_patches1;
            inv_mask_data = inv_mask_patches1;
        else
            input_data = cat(4, input_data, input_patches1);
            label_data = cat(4, label_data, label_patches1);
            interlaced_data = cat(4, interlaced_data, interlaced_patches1);
            deinterlaced_data = cat(4, deinterlaced_data, deinterlaced_patches1);
            inv_mask_data = cat(4, inv_mask_data, inv_mask_patches1);
        end
    end
    
    %{
    for i = size(input_data, 4):-1:1
        tmp = abs(input_data(:, :, 1, i) - input_data(:, :, 3, i));
        diffs(i) = sum(tmp(:));
    end
    
    %indexes = diffs <= floor(mean(diffs));
    indexes = diffs > 28.4727;
    input_data = input_data(:, :, :, indexes(:));
    label_data = label_data(:, :, :, indexes(:));
    interlaced_data = interlaced_data(:, :, :, indexes(:));
    deinterlaced_data = deinterlaced_data(:, :, :, indexes(:));
    inv_mask_data = inv_mask_data(:, :, :, indexes(:));
    %}
    
    save2hdf5(savepath, chunksz, input_data, label_data, interlaced_data, deinterlaced_data, inv_mask_data);
end
