function [] = generate_data()
    clear;
    close all;
    
    %% Settings
    patch_method = 1;
    self_learning = 0;
    is_train_data = 0;
    input_channels = 3;
    testFramesCnt = 100;
    
    if patch_method == 1
        window = [30, 30];
    else
        window = 3;
    end

    if is_train_data
        folder = 'Train';
        savepath = 'train.h5';
        chunksz = 2048;
    else
        folder = 'Test';
        savepath = 'test.h5';
        chunksz = 2048;
    end
    
    filepaths = dir(fullfile(folder,'*.avi'));
    
    if patch_method == 1
        gen_patch2patch_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, input_channels, self_learning);
    else
        gen_patch2pixel_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, input_channels, self_learning);
    end
end

function [] = gen_patch2pixel_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, input_channels, self_learning)
    for i = 1:length(filepaths)
        frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
        
        if self_learning
            [resizeds1, resizeds2] = interlaced_resize(frames);        
            [input_patches1, label_patches1] = patch2pixel(resizeds1, window, input_channels);
            [input_patches2, label_patches2] = patch2pixel(resizeds2, window, input_channels);
                                    
            if i == 1
                input_data = cat(4, input_patches1, input_patches2);
                label_data = cat(4, label_patches1, label_patches2);
            else
                input_data = cat(4, input_data, input_patches1, input_patches2);
                label_data = cat(4, label_data, label_patches1, label_patches2);
            end
        else      
            [input_patches1, label_patches1] = patch2pixel(frames, window, input_channels);
        
            if i == 1
                input_data = input_patches1;
                label_data = label_patches1;
            else
                input_data = cat(4, input_data, input_patches1);
                label_data = cat(4, label_data, label_patches1);
            end
        end
    end
    
    save2hdf5(savepath, chunksz, input_data, label_data);
end

function [] = gen_patch2patch_data(folder, filepaths, savepath, chunksz, testFramesCnt, window, input_channels, self_learning)
    for i = 1:length(filepaths)
        frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
        
        if self_learning
            [resizeds1, resizeds2] = interlaced_resize(frames);
            [input_patches1, label_patches1, interlaced_patches1, deinterlaced_patches1, inv_mask_patches1] = patch2patch(resizeds1, window, input_channels);
            [input_patches2, label_patches2, interlaced_patches2, deinterlaced_patches2, inv_mask_patches2] = patch2patch(resizeds2, window, input_channels);
        
            if i == 1
                input_data = cat(4, input_patches1, input_patches2);
                label_data = cat(4, label_patches1, label_patches2);
                interlaced_data = cat(4, interlaced_patches1, interlaced_patches2);
                deinterlaced_data = cat(4, deinterlaced_patches1, deinterlaced_patches2);
                inv_mask_data = cat(4, inv_mask_patches1, inv_mask_patches2);
            else
                input_data = cat(4, input_data, input_patches1, input_patches2);
                label_data = cat(4, label_data, label_patches1, label_patches2);
                interlaced_data = cat(4, interlaced_data, interlaced_patches1, interlaced_patches2);
                deinterlaced_data = cat(4, deinterlaced_data, deinterlaced_patches1, deinterlaced_patches2);
                inv_mask_data = cat(4, inv_mask_data, inv_mask_patches1, inv_mask_patches2);
            end
        else      
            [input_patches1, label_patches1, interlaced_patches1, deinterlaced_patches1, inv_mask_patches1] = patch2patch(frames, window, input_channels);
        
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
