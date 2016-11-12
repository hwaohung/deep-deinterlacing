filename = 'C:\Users\Johnny\Desktop\碩士論文\data\test\akiyo_cif';
filename = [filename, '.avi'];

folder = 'Test';
filepaths = dir(fullfile(folder, '*.avi'));
testFramesCnt = 300;

gnd_frames = get_video_frames(filename, testFramesCnt);
% Base
frames1 = deinterlace_video(gnd_frames, testFramesCnt);
% Final
frames2 = DeepDeinterlacing(frames1, 100000);
% Naive deinterlace

clear a b c diff;
for i = 1:size(gnd_frames, 3)
    if mod(i, 2)
        s_i = 2;
        d_i = 1;
    else
        s_i = 1;
        d_i = 2;
    end
    
    %a(:, :, i) = gnd_frames(s_i:2:end, :, i) - frames1(s_i:2:end, :, i);
    %b(:, :, i) = gnd_frames(s_i:2:end, :, i) - frames2(s_i:2:end, :, i);
    a = gnd_frames(:, :, i) - frames1(:, :, i);
    b = gnd_frames(:, :, i) - frames2(:, :, i);
    
    temp = deinterlace(gnd_frames(:, :, i), mod(i, 2));
    %c(:, :, i) = gnd_frames(s_i:2:end, :, i) - temp(s_i:2:end, :);
    c = gnd_frames(:, :, i) - temp;
    
    %{
    for row = s_i:2:size(frames2, 1)
        for col = 1:size(frames2, 2)
            if 
        end
    end
    %}
    
    %{
    if i >= 2 && i <= testFramesCnt-2
        diff = gnd_frames(d_i:2:end, :, i-1) - gnd_frames(d_i:2:end, :, i+1);
    end
    %}
    
    %{
    if i == 1
        
    elseif i == testFramesCnt
        tmp1 = (gnd_frames(:, :, i-1) - gnd_frames(:, :, i+1)) .^ 2;
    else
        tmp1 = (gnd_frames(:, :, i-1) - gnd_frames(:, :, i+1)) .^ 2;
    end
    
    mask = [0, 0, 0; 1, 1, 1; 0, 0, 0];
    tmp1 = conv2(tmp1, mask, 'same');
    diff(:, :, i) = tmp1(s_i:2:end, :);
    %}
    
    %{
    j1 = abs(b) - abs(a);
    tmp1 = frames2(:, :, i);
    tmp2 = frames1(:, :, i);
    tmp1(j1>0) = tmp2(j1>0);
    %}
    
    j1 = abs(b) - abs(c);
    tmp1 = frames2(:, :, i);
    tmp2 = temp;
    tmp1(j1>0) = tmp2(j1>0);
    
    frames2(:, :, i) = tmp1;
    %psnrs1(i) = compute_psnr(gnd_frames(:, :, i), frames1(:, :, i));
    psnrs2(i) = compute_psnr(gnd_frames(:, :, i), frames2(:, :, i));
end

mean(psnrs2)
j1 = abs(b) - abs(a);

%psnr = compute_psnr(gnd_frames, frames2);
%frames2(j1>0) = frames1(j1>0);
%psnr = compute_psnr(gnd_frames, frames2);

j1 = j1(j1 > 0);

j2 = abs(b) - abs(c);
j2 = j2(j2 > 0);
return;

a = psnrs1 - psnrs2;
a = a(:, a>0);

disp(a);

return;

sum(psnrs1(:))/size(psnrs1, 2)
sum(psnrs2(:))/size(psnrs2, 2)

return;

filename = 'C:\Users\Johnny\Desktop\碩士論文\data\test\stefan_sif';
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
    frames1(:, :, i) = frame2;
    
    if i == 300
        break;
    end
end

sum(psnrs(:))/size(psnrs, 2)