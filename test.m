filename = 'C:\Users\Johnny\Desktop\ºÓ¤h½×¤å\data\test\bowing_cif';
v1 = VideoReader(strcat(filename, '_original', '.avi'));
v2 = VideoReader(strcat(filename, '_interlaced', '.avi'));

i = 0;
while hasFrame(v1)
    i = i + 1;
    frame1 = readFrame(v1);
    %frame2 = readFrame(v2);
    [frame2] = interlace(frame1, mod(i, 2));
        
    frames(:, :, i) = frame2;
end


[resizeds1, resizeds2] = resize_frames(frames);
for i = 1:size(resizeds1, 3)
    figure(1), imshow(resizeds1(:, :, i)); title('resizeds1');
    figure(2), imshow(resizeds2(:, :, i)); title('resizeds2');
    pause;
end

%hdint = vision.Deinterlacer;
%hdint = vision.Deinterlacer('Method', 'Linear interpolation', 'TransposedInput', false);
%y = step(hdint, frame2);