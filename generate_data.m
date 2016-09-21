clear;close all;

%% settings
is_train_data = 0;
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

%% initialization
input_data = zeros(size_input, size_input, 3, 1);
label_data = zeros(size_label, size_label, 1, 1);
interlaced_data = zeros(size_input, size_input, 1, 1);
deinterlaced_data = zeros(size_input, size_input, 1, 1);
inv_mask_data = zeros(size_input, size_input, 1, 1);
count = 0;

%% Generate data
for i = 1:length(filepaths)   
    %% Get frames, interlaced_fields, inv_masks, deinterlaced_fields
    v = VideoReader(fullfile(folder, filepaths(i).name));
    hei = v.height;
    wid = v.width;
    
    for frameCnt = 1:testFramesCnt
        if ~hasFrame(v)
            break;
        end
        
        frame = readFrame(v);
        % gray image only
        if size(frame, 3) > 1
            frame = rgb2gray(frame);
        end
        
        % image even * even size(why)
        frame = modcrop(frame, 2);
        
        [interlaced_field, inv_mask] = interlace(frame, mod(frameCnt, 2));
        deinterlaced_field = deinterlace(interlaced_field, mod(frameCnt, 2));
        
        frames(:, :, frameCnt) = im2double(frame);
        interlaced_fields(:, :, frameCnt) = im2double(interlaced_field);
        inv_masks(:, :, frameCnt) = im2double(inv_mask);
        deinterlaced_fields(:, :, frameCnt) = im2double(deinterlaced_field);
    end
    
    %% Generate data pacth
    for frameCnt = 1:testFramesCnt         
        % Get prev, post field
        if frameCnt == 1
            prev = deinterlaced_fields(:, :, frameCnt);
            post = deinterlaced_fields(:, :, frameCnt+1);
        elseif frameCnt == testFramesCnt
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
            return;
        end
        %}
        
        for x = 1 : stride : hei-size_input+1
            for y = 1 :stride : wid-size_input+1
                count = count + 1;
                input_data(:, :, :, count) = input_full(x:x+size_input-1, y:y+size_input-1, 1:3);
                label_data(:, :, :, count) = label_full(x:x+size_label-1, y:y+size_label-1, 1);
                interlaced_data(:, :, :, count) = interlace_full(x:x+size_input-1, y:y+size_input-1, 1);
                deinterlaced_data(:, :, :, count) = deinterlace_full(x:x+size_input-1, y:y+size_input-1, 1);
                inv_mask_data(:, :, :, count) = inv_mask_full(x:x+size_input-1, y:y+size_input-1, 1);
            end
        end
    end
    
    clear v frames fields inv_masks deinterlaced_fields;
end

%% Data order rearrange
order = randperm(count);
input_data = input_data(:, :, :, order); 
label_data = label_data(:, :, :, order); 
interlaced_data = interlaced_data(:, :, :, order);
deinterlaced_data = deinterlaced_data(:, :, :, order);
inv_mask_data = inv_mask_data(:, :, :, order);

%% writing to HDF5
%chunksz = 128;
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
