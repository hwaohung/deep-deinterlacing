clear;close all;

%% settings
is_train_data = 1;
size_input = 31;
size_label = 31;
input_channel = 1;
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
data = zeros(size_input, size_input, input_channel, 1);
datai = zeros(size_input, size_input, input_channel, 1);
inv_mask = zeros(size_input, size_input, input_channel, 1);
label = zeros(size_label, size_label, input_channel, 1);
count = 0;

%% generate data
for i = 1 : length(filepaths)
    v = VideoReader(fullfile(folder, filepaths(i).name));
    
    frameCnt = 0;
    while hasFrame(v)
        image = readFrame(v);
        frameCnt = frameCnt + 1;
        
        [hei, wid, c] = size(image);
        if c > 1
            image = rgb2gray(image);
        end
        % image even * even size(why)
        image = modcrop(image, 2);
        
        [im_input, im_mask] = interlace(image, mod(frameCnt, 2));

        im_input = im2double(im_input);
        im_input_i = im2double(deinterlace(im_input, mod(frameCnt, 2)));
        im_mask = im2double(im_mask);
        im_label = im2double(image);
        
        % Test code for check image is ok
        %{
        if frameCnt == 2
            imshow(im_input); title('Interlace Image');
            figure, imshow(im_input_i); title('De-interlace Image');
            figure, imshow(im_mask); title('Mask Image');
            figure, imshow(im_label); title('Label Image');
            return;
        end
        %}
        
        for x = 1 : stride : hei-size_input+1
            for y = 1 :stride : wid-size_input+1
                count = count + 1;
                data(:, :, :, count) = im_input(x : x+size_input-1, y : y+size_input-1, 1:input_channel);
                datai(:, :, :, count) = im_input_i(x : x+size_input-1, y : y+size_input-1, 1:input_channel);
                label(:, :, :, count) = im_label(x : x+size_label-1, y : y+size_label-1, 1:input_channel);
                inv_mask(:, :, :, count) = im_mask(x : x+size_input-1, y : y+size_input-1, 1:input_channel);
            end
        end
        
        if frameCnt == 30
            break;
        end
    end
end

order = randperm(count);
data = data(:, :, :, order);
datai = datai(:, :, :, order);
inv_mask = inv_mask(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
%chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read = (batchno-1) * chunksz;

    batchdata = data(:, :, :, last_read+1:last_read+chunksz); 
    batchdatai = datai(:, :, :, last_read+1:last_read+chunksz); 
    batchdata_inv_mask = inv_mask(:, :, :, last_read+1:last_read+chunksz); 
    batchlabs = label(:, :, :, last_read+1:last_read+chunksz);

    startloc = struct('dat', [1,1,1,totalct+1],...
                      'dat_i', [1,1,1,totalct+1],...
                      'inv_mask', [1,1,1,totalct+1],...
                      'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchdatai, batchdata_inv_mask, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
