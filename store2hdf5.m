function [curr_dat_sz, curr_lab_sz] = store2hdf5(filename, input_data, label_data, interlaced_data, deinterlaced_data, inv_mask_data, create, startloc, chunksz)  
  % *data* is W*H*C*N matrix of images should be normalized (e.g. to lie between 0 and 1) beforehand
  % *label* is D*N matrix of labels (D labels per sample) 
  % *create* [0/1] specifies whether to create file newly or to append to previously created file, useful to store information in batches when a dataset is too big to be held in memory  (default: 1)
  % *startloc* (point at which to start writing data). By default, 
  % if create=1 (create mode), startloc.data=[1 1 1 1], and startloc.lab=[1 1]; 
  % if create=0 (append mode), startloc.data=[1 1 1 K+1], and startloc.lab = [1 K+1]; where K is the current number of samples stored in the HDF
  % chunksz (used only in create mode), specifies number of samples to be stored per chunk (see HDF5 documentation on chunking) for creating HDF5 files with unbounded maximum size - TLDR; higher chunk sizes allow faster read-write operations 
    
  % verify that format is right
    input_data_dims = size(input_data);
    label_data_dims = size(label_data);
    interlaced_data_dims = size(interlaced_data);
    deinterlaced_data_dims = size(deinterlaced_data);
    inv_mask_dims = size(inv_mask_data);
  
    num_samples = input_data_dims(end);
    assert(label_data_dims(end)==num_samples, 'Number of samples should be matched between data and labels');
  
    if ~exist('create','var')
        create = true;
    end
  
    if create
        %fprintf('Creating dataset with %d samples\n', num_samples);
        if ~exist('chunksz', 'var')
            chunksz = 1000;
        end
        
        if exist(filename, 'file')
            fprintf('Warning: replacing existing file %s \n', filename);
            delete(filename);
        end
    
        h5create(filename, '/input', [input_data_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [input_data_dims(1:end-1) chunksz]); % width, height, channels, number 
        h5create(filename, '/label', [label_data_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [label_data_dims(1:end-1) chunksz]); % width, height, channels, number 
        h5create(filename, '/interlace', [interlaced_data_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [interlaced_data_dims(1:end-1) chunksz]); % width, height, channels, number 
        h5create(filename, '/deinterlace', [deinterlaced_data_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [deinterlaced_data_dims(1:end-1) chunksz]); % width, height, channels, number 
        h5create(filename, '/inv-mask', [inv_mask_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [inv_mask_dims(1:end-1) chunksz]); % width, height, channels, number
            
        if ~exist('startloc','var')
            startloc.input_data = [ones(1,length(input_data_dims)-1), 1];
            startloc.label_data = [ones(1,length(label_data_dims)-1), 1];
            startloc.interlaced_data = [ones(1,length(interlaced_data_dims)-1), 1];
            startloc.deinterlaced_data = [ones(1,length(deinterlaced_data_dims)-1), 1];
            startloc.inv_mask_data = [ones(1,length(inv_mask_dims)-1), 1];      
        end 
    else  % append mode
        if ~exist('startloc','var')
            info = h5info(filename);
        
            prev_input_data_sz = info.Datasets(1).Dataspace.Size;
            prev_label_data_sz = info.Datasets(2).Dataspace.Size;
            prev_interlaced_data_sz = info.Datasets(3).Dataspace.Size;
            prev_deinterlaced_data_sz = info.Datasets(4).Dataspace.Size;
            prev_inv_mask_data_sz = info.Datasets(5).Dataspace.Size;
        
            assert(prev_input_data_sz(1:end-1)==input_data_dims(1:end-1), 'Input data dimensions must match existing dimensions in dataset');
            assert(prev_label_data_sz(1:end-1)==label_data_dims(1:end-1), 'Label data dimensions must match existing dimensions in dataset');
            assert(prev_interlaced_data_sz(1:end-1)==interlaced_data_dims(1:end-1), 'Interlaced data dimensions must match existing dimensions in dataset');
            assert(prev_deinterlaced_data_sz(1:end-1)==deinterlaced_data_dims(1:end-1), 'Deinterlaced data dimensions must match existing dimensions in dataset');
            assert(prev_inv_mask_data_sz(1:end-1)==inv_mask_dims(1:end-1), 'Inv-mask data dimensions must match existing dimensions in dataset');
                
            startloc.input_data = [ones(1,length(input_data_dims)-1), prev_input_data_sz(end)+1];
            startloc.label_data = [ones(1,length(label_data_dims)-1), prev_label_data_sz(end)+1];
            startloc.interlaced_data = [ones(1,length(interlaced_data_dims)-1), prev_interlaced_data_sz(end)+1];
            startloc.deinterlaced_data = [ones(1,length(deinterlaced_data_dims)-1), prev_deinterlaced_data_sz(end)+1];
            startloc.inv_mask_data = [ones(1,length(inv_mask_dims)-1), prev_inv_mask_data_sz(end)+1];
        end
    end
    
    if ~isempty(input_data)
        h5write(filename, '/input', single(input_data), startloc.input_data, size(input_data));  
        h5write(filename, '/label', single(label_data), startloc.label_data, size(label_data));  
        h5write(filename, '/interlace', single(interlaced_data), startloc.interlaced_data, size(interlaced_data));
        h5write(filename, '/deinterlace', single(deinterlaced_data), startloc.deinterlaced_data, size(deinterlaced_data));
        h5write(filename, '/inv-mask', single(inv_mask_data), startloc.inv_mask_data, size(inv_mask_data));    
    end

    if nargout
        info = h5info(filename);
        curr_dat_sz = info.Datasets(1).Dataspace.Size;
        curr_lab_sz = info.Datasets(2).Dataspace.Size;
    end
end
