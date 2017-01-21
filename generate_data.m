function [] = generate_data()
    clear;
    close all;
    
    %% Settings
    patch_method = 1;
    is_train_data = 0;
    testFramesCnt = 100;
    
    % stride, window(1) must be even(even shift for sure the same parity)
    if patch_method == 1
        window = [30, 30];
        stride = 30;
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
        chunksz = 64;
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
        
        [input_patches1, label_patches1, deinterlaced_patches1, flags1, eachCnt] = patch2patch(frames, window, stride);
        
        if i == 1
        	input_data = input_patches1;
            label_data = label_patches1;
            deinterlaced_data = deinterlaced_patches1;
            flag_data = flags1;
        else
            input_data = cat(4, input_data, input_patches1);
            label_data = cat(4, label_data, label_patches1);
            deinterlaced_data = cat(4, deinterlaced_data, deinterlaced_patches1);
            flag_data = cat(4, flag_data, flags1);
        end
    end
    
    %a = round(flags1(:));
    %histogram(a);
    
    % Threshold
    indexes = flag_data < Var.T;
    indexes = indexes(:);
    
    save2hdf5([savepath '.static'], chunksz, input_data(:, :, :, indexes), label_data(:, :, :, indexes), deinterlaced_data(:, :, :, indexes));
    save2hdf5([savepath '.dynamic'], chunksz, input_data(:, :, :, ~indexes), label_data(:, :, :, ~indexes), deinterlaced_data(:, :, :, ~indexes));
end
