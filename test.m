
hinfo = hdf5info('train.h5');
%tmp = hinfo.GroupHierarchy.Groups(1);
tmp = hinfo.GroupHierarchy;
dset = hdf5read(tmp.Datasets(2));
return;


filename = 'C:\Users\Johnny\Desktop\ºÓ¤h½×¤å\data\test\bowing_cif';
v1 = VideoReader(strcat(filename, '_original', '.avi'));
v2 = VideoReader(strcat(filename, '_interlaced', '.avi'));

i = 0;
while hasFrame(v2)
    i = i + 1;
    frame1 = readFrame(v1);
    frame2 = readFrame(v2);
    
    %[frame1, mask] = interlace(frame1, mod(i, 2) == 0);
    
    y = deinterlace(frame2, mod(i, 2) == 0);
    if i == 2
        break;
    end
end

%hdint = vision.Deinterlacer;
%hdint = vision.Deinterlacer('Method', 'Linear interpolation', 'TransposedInput', false);
%y = step(hdint, frame2);

imshow(frame1); title('Original Image');
figure, imshow(y); title('Image after deinterlacing');