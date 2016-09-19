function [curr_dat_sz, curr_lab_sz] = store2hdf5(filename, data, datai, inv_mask, labels, create, startloc, chunksz)  
  % *data* is W*H*C*N matrix of images should be normalized (e.g. to lie between 0 and 1) beforehand
  % *label* is D*N matrix of labels (D labels per sample) 
  % *create* [0/1] specifies whether to create file newly or to append to previously created file, useful to store information in batches when a dataset is too big to be held in memory  (default: 1)
  % *startloc* (point at which to start writing data). By default, 
  % if create=1 (create mode), startloc.data=[1 1 1 1], and startloc.lab=[1 1]; 
  % if create=0 (append mode), startloc.data=[1 1 1 K+1], and startloc.lab = [1 K+1]; where K is the current number of samples stored in the HDF
  % chunksz (used only in create mode), specifies number of samples to be stored per chunk (see HDF5 documentation on chunking) for creating HDF5 files with unbounded maximum size - TLDR; higher chunk sizes allow faster read-write operations 

  % verify that format is right
  dat_dims=size(data);
  dat_i_dims=size(datai);
  inv_mask_dims=size(inv_mask);
  lab_dims=size(labels);
  num_samples=dat_dims(end);

  assert(lab_dims(end)==num_samples, 'Number of samples should be matched between data and labels');

  if ~exist('create','var')
    create=true;
  end

  
  if create
    %fprintf('Creating dataset with %d samples\n', num_samples);
    if ~exist('chunksz', 'var')
      chunksz=1000;
    end
    if exist(filename, 'file')
      fprintf('Warning: replacing existing file %s \n', filename);
      delete(filename);
    end      
    h5create(filename, '/data', [dat_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [dat_dims(1:end-1) chunksz]); % width, height, channels, number 
    h5create(filename, '/data-i', [dat_i_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [dat_i_dims(1:end-1) chunksz]); % width, height, channels, number 
    h5create(filename, '/inv-mask', [inv_mask_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [inv_mask_dims(1:end-1) chunksz]); % width, height, channels, number 
    h5create(filename, '/label', [lab_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [lab_dims(1:end-1) chunksz]); % width, height, channels, number 
    if ~exist('startloc','var') 
      startloc.dat=[ones(1,length(dat_dims)-1), 1];
      startloc.dat_i=[ones(1,length(dat_i_dims)-1), 1];
      startloc.inv_mask=[ones(1,length(inv_mask_dims)-1), 1];
      startloc.lab=[ones(1,length(lab_dims)-1), 1];
    end 
  else  % append mode
    if ~exist('startloc','var')
      info=h5info(filename);
      prev_dat_sz=info.Datasets(1).Dataspace.Size;
      prev_dat_i_sz=info.Datasets(2).Dataspace.Size;
      prev_inv_mask_sz=info.Datasets(3).Dataspace.Size;
      prev_lab_sz=info.Datasets(4).Dataspace.Size;
      assert(prev_dat_sz(1:end-1)==dat_dims(1:end-1), 'Data dimensions must match existing dimensions in dataset');
      assert(prev_dat_i_sz(1:end-1)==dat_i_dims(1:end-1), 'Datai dimensions must match existing dimensions in dataset');
      assert(prev_inv_mask_sz(1:end-1)==inv_mask_dims(1:end-1), 'Datai dimensions must match existing dimensions in dataset');
      assert(prev_lab_sz(1:end-1)==lab_dims(1:end-1), 'Label dimensions must match existing dimensions in dataset');
      startloc.dat=[ones(1,length(dat_dims)-1), prev_dat_sz(end)+1];
      startloc.dat_i=[ones(1,length(dat_i_dims)-1), prev_dat_i_sz(end)+1];
      startloc.inv_mask=[ones(1,length(inv_mask_dims)-1), prev_inv_mask_sz(end)+1];
      startloc.lab=[ones(1,length(lab_dims)-1), prev_lab_sz(end)+1];
    end
  end

  if ~isempty(data)
    h5write(filename, '/data', single(data), startloc.dat, size(data));
    h5write(filename, '/data-i', single(datai), startloc.dat_i, size(datai));
    h5write(filename, '/inv-mask', single(inv_mask), startloc.inv_mask, size(inv_mask));
    h5write(filename, '/label', single(labels), startloc.lab, size(labels));  
  end

  if nargout
    info=h5info(filename);
    curr_dat_sz=info.Datasets(1).Dataspace.Size;
    curr_lab_sz=info.Datasets(4).Dataspace.Size;
  end
end
