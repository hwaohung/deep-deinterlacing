function [] = generate_data()
    clear;
    close all;
    
    %% Settings
    is_train_data = 1;
    testFramesCnt = 30; 

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
    
    gen_new_patch2pixel_data(folder, filepaths, savepath, chunksz, testFramesCnt);
end

function [] = gen_new_patch2pixel_data(folder, filepaths, savepath, chunksz, testFramesCnt)
    for i = 1:length(filepaths)
        frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
        
        [input_patches1, label_patches1] = new_patch2pixel(frames);
        
        % Remove condition(for calssify data usage)
        indexes = eval_t(input_patches1, 0.027, 1);
        input_patches1 = input_patches1(:, :, :, indexes);
        label_patches1 = label_patches1(:, :, :, indexes);
        
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
