function [] = save2hdf5(savepath, chunksz, input_data, label_data, interlaced_data, deinterlaced_data, inv_mask_data)
    %% Data order rearrange
    count = size(input_data, 4);
    order = randperm(count);
    input_data = input_data(:, :, :, order);
    label_data = label_data(:, :, :, order);
    if exist('interlaced_data','var')
        interlaced_data = interlaced_data(:, :, :, order);
        deinterlaced_data = deinterlaced_data(:, :, :, order);
        inv_mask_data = inv_mask_data(:, :, :, order);
    end

    created_flag = false;
    totalct = 0;

    for batchno = 1:floor(count/chunksz)
        last_read = (batchno-1) * chunksz;

        b_input_data = input_data(:, :, :, last_read+1:last_read+chunksz);
        b_label_data = label_data(:, :, :, last_read+1:last_read+chunksz);
        
        if exist('interlaced_data','var')
            b_interlaced_data = interlaced_data(:, :, :, last_read+1:last_read+chunksz); 
            b_deinterlaced_data = deinterlaced_data(:, :, :, last_read+1:last_read+chunksz); 
            b_inv_mask_data = inv_mask_data(:, :, :, last_read+1:last_read+chunksz);
            
            startloc = struct('input_data', [1,1,1,totalct+1], ...
                              'label_data', [1,1,1,totalct+1], ...
                              'interlaced_data', [1,1,1,totalct+1], ...
                              'deinterlaced_data', [1,1,1,totalct+1], ...
                              'inv_mask_data', [1,1,1,totalct+1]);
            curr_dat_sz = write_hdf5(savepath, chunksz, ~created_flag, startloc, b_input_data, b_label_data, b_interlaced_data, b_deinterlaced_data, b_inv_mask_data);
        else
            startloc = struct('input_data', [1, 1, 1, totalct+1], ...
                              'label_data', [1, 1, 1, totalct+1]);
            curr_dat_sz = write_hdf5(savepath, chunksz, ~created_flag, startloc, b_input_data, b_label_data); 
        end
    
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    
    h5disp(savepath);
end

function [curr_dat_sz, curr_lab_sz] = write_hdf5(filename, chunksz, create, startloc, input_data, label_data, interlaced_data, deinterlaced_data, inv_mask_data) 
  % *data* is W*H*C*N matrix of images should be normalized (e.g. to lie between 0 and 1) beforehand
  % *label* is D*N matrix of labels (D labels per sample) 
  % *create* [0/1] specifies whether to create file newly or to append to previously created file, useful to store information in batches when a dataset is too big to be held in memory  (default: 1)
  % *startloc* (point at which to start writing data). By default, 
  % if create=1 (create mode), startloc.data=[1 1 1 1], and startloc.lab=[1 1]; 
  % if create=0 (append mode), startloc.data=[1 1 1 K+1], and startloc.lab = [1 K+1]; where K is the current number of samples stored in the HDF
  % chunksz (used only in create mode), specifies number of samples to be stored per chunk (see HDF5 documentation on chunking) for creating HDF5 files with unbounded maximum size - TLDR; higher chunk sizes allow faster read-write operations 
    
    input_data_dims = size(input_data);
    label_data_dims = size(label_data);
    % Use interlaced_data to classify patch or pixel method
    if exist('interlaced_data','var')
        interlaced_data_dims = size(interlaced_data);
        deinterlaced_data_dims = size(deinterlaced_data);
        inv_mask_dims = size(inv_mask_data);
    end
  
    num_samples = input_data_dims(end);
    assert(label_data_dims(end)==num_samples, 'Number of samples should be matched between data and labels');
  
    if ~exist('create','var')
        create = true;
    end
  
    if create
        %fprintf('Creating dataset with %d samples\n', num_samples);
        if exist(filename, 'file')
            fprintf('Warning: replacing existing file %s \n', filename);
            delete(filename);
        end
    
        h5create(filename, '/input', [input_data_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [input_data_dims(1:end-1) chunksz]); % width, height, channels, number 
        h5create(filename, '/label', [label_data_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [label_data_dims(1:end-1) chunksz]); % width, height, channels, number         
        if exist('interlaced_data','var')
            h5create(filename, '/interlace', [interlaced_data_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [interlaced_data_dims(1:end-1) chunksz]); % width, height, channels, number 
            h5create(filename, '/deinterlace', [deinterlaced_data_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [deinterlaced_data_dims(1:end-1) chunksz]); % width, height, channels, number 
            h5create(filename, '/inv-mask', [inv_mask_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [inv_mask_dims(1:end-1) chunksz]); % width, height, channels, number
        end
        
        if ~exist('startloc','var')
            startloc.input_data = [ones(1,length(input_data_dims)-1), 1];
            startloc.label_data = [ones(1,length(label_data_dims)-1), 1];  
            if exist('interlaced_data','var')
                startloc.interlaced_data = [ones(1,length(interlaced_data_dims)-1), 1];
                startloc.deinterlaced_data = [ones(1,length(deinterlaced_data_dims)-1), 1];
                startloc.inv_mask_data = [ones(1,length(inv_mask_dims)-1), 1];
            end
        end 
    else  % append mode
        if ~exist('startloc','var')
            info = h5info(filename);
        
            prev_input_data_sz = info.Datasets(1).Dataspace.Size;
            prev_label_data_sz = info.Datasets(2).Dataspace.Size;
            
            assert(prev_input_data_sz(1:end-1)==input_data_dims(1:end-1), 'Input data dimensions must match existing dimensions in dataset');
            assert(prev_label_data_sz(1:end-1)==label_data_dims(1:end-1), 'Label data dimensions must match existing dimensions in dataset');
                
            startloc.input_data = [ones(1,length(input_data_dims)-1), prev_input_data_sz(end)+1];
            startloc.label_data = [ones(1,length(label_data_dims)-1), prev_label_data_sz(end)+1];
            
            if exist('interlaced_data','var')
                prev_interlaced_data_sz = info.Datasets(3).Dataspace.Size;
                prev_deinterlaced_data_sz = info.Datasets(4).Dataspace.Size;
                prev_inv_mask_data_sz = info.Datasets(5).Dataspace.Size;
                
                assert(prev_interlaced_data_sz(1:end-1)==interlaced_data_dims(1:end-1), 'Interlaced data dimensions must match existing dimensions in dataset');
                assert(prev_deinterlaced_data_sz(1:end-1)==deinterlaced_data_dims(1:end-1), 'Deinterlaced data dimensions must match existing dimensions in dataset');
                assert(prev_inv_mask_data_sz(1:end-1)==inv_mask_dims(1:end-1), 'Inv-mask data dimensions must match existing dimensions in dataset');
                
                startloc.interlaced_data = [ones(1,length(interlaced_data_dims)-1), prev_interlaced_data_sz(end)+1];
                startloc.deinterlaced_data = [ones(1,length(deinterlaced_data_dims)-1), prev_deinterlaced_data_sz(end)+1];
                startloc.inv_mask_data = [ones(1,length(inv_mask_dims)-1), prev_inv_mask_data_sz(end)+1];
            end
        end
    end
    
    if ~isempty(input_data)
        h5write(filename, '/input', single(input_data), startloc.input_data, size(input_data));  
        h5write(filename, '/label', single(label_data), startloc.label_data, size(label_data));
        if exist('interlaced_data','var')
            h5write(filename, '/interlace', single(interlaced_data), startloc.interlaced_data, size(interlaced_data));
            h5write(filename, '/deinterlace', single(deinterlaced_data), startloc.deinterlaced_data, size(deinterlaced_data));
            h5write(filename, '/inv-mask', single(inv_mask_data), startloc.inv_mask_data, size(inv_mask_data));    
        end
    end

    if nargout
        info = h5info(filename);
        curr_dat_sz = info.Datasets(1).Dataspace.Size;
        curr_lab_sz = info.Datasets(2).Dataspace.Size;
    end
end
