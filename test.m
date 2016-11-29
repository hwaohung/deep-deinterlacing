clear all;

%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\gray\akiyo_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\gray\coastguard_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\gray\container_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\gray\foreman_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\gray\flower_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\gray\hall_monitor_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\gray\mother_daughter_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\gray\stefan_cif';


%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\akiyo_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\coastguard_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\container_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\foreman_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\flower_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\hall_monitor_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\mother_daughter_cif';
filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\stefan_cif';
filename = [filename, '.avi'];

folder = 'Test';
filepaths = dir(fullfile(folder, '*.avi'));
testFramesCnt = 90;

gnd_frames = get_video_frames(filename, testFramesCnt);

tic;
%frames1 = deinterlace(gnd_frames, 1);
frames1 = self_validation(gnd_frames, 4, 1);
frames2 = frames1;
disp(toc);

%{
tic;
%frames2 = self_validation(gnd_frames, 6, 1);
%frames2 = DeepTemp(gnd_frames, 100000, 1, 1);
disp(toc);
%}

for i = 5:size(gnd_frames, 4)
    imshow(gnd_frames(:, :, :, i));
    psnrs1(i) = psnr(frames1(:, :, :, i), gnd_frames(:, :, :, i));
    psnrs2(i) = psnr(frames2(:, :, :, i), gnd_frames(:, :, :, i));
    
    %ssims1(i) = ssim(frames1(:, :, :, i), gnd_frames(:, :, :, i));
    %ssims2(i) = ssim(frames2(:, :, :, i), gnd_frames(:, :, :, i));
end

disp([mean(psnrs1), mean(psnrs2)]);
%disp([mean(ssims1), mean(ssims2)]);