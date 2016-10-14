filename = 'C:\Users\Johnny\Desktop\�Ӥh�פ�\data\test\akiyo_cif';
v1 = VideoReader(strcat(filename, '_original', '.avi'));

i = 0;
while hasFrame(v1)
    i = i + 1;
    frame1 = readFrame(v1);
    frame2 = deinterlace(frame1, mod(i, 2));
    
    psnrs(i) = compute_psnr(frame1, frame2);
    frames(:, :, i) = frame2;
    
    %figure(1), imshow(frame1); title('Original');
    figure(2), imshow(frame2); title('Deinterlaced');
end

sum(psnrs(:))/size(psnrs, 2)