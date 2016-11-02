filename = 'C:\Users\Johnny\Desktop\ºÓ¤h½×¤å\data\test\container_cif';
v1 = VideoReader(strcat(filename, '.avi'));

i = 0;
while hasFrame(v1)
    i = i + 1;
    frame1 = readFrame(v1);
    tic;
    frame2 = deinterlace(frame1, mod(i, 2));
    run_time = toc;
    disp(run_time);
    
    psnrs(i) = compute_psnr(frame1, frame2);
    frames(:, :, i) = frame2;
    
    if i == 30
        break;
    end
    
    %figure(1), imshow(frame1); title('Original');
    %figure(2), imshow(frame2); title('Deinterlaced');
end

sum(psnrs(:))/size(psnrs, 2)