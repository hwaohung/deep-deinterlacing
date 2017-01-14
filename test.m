%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\akiyo_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\coastguard_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\container_cif';
filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\foreman_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\flower_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\hall_monitor_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\mother_daughter_cif';
%filename = 'C:\Users\Johnny\Desktop\Master_thesis\data\test\stefan_cif';
filename = [filename, '.avi'];

folder = 'Test';
filepaths = dir(fullfile(folder, '*.avi'));
testFramesCnt = 100;

frames = get_video_frames(filename, testFramesCnt);

Kps = zeros(1);
Kns = zeros(1);
Alls = zeros(1);
for fCnt = 1:testFramesCnt
    if fCnt == 1
            prev2 = frames(:, :, :, fCnt+2);
            prev1 = frames(:, :, :, fCnt+1);
            post1 = frames(:, :, :, fCnt+1);
            post2 = frames(:, :, :, fCnt+2);
    elseif fCnt == 2
            prev2 = frames(:, :, :, fCnt+2);
            prev1 = frames(:, :, :, fCnt-1);
            post1 = frames(:, :, :, fCnt+1);
            post2 = frames(:, :, :, fCnt+2);
    elseif fCnt == testFramesCnt-1
            prev2 = frames(:, :, :, fCnt-2);
            prev1 = frames(:, :, :, fCnt-1);
            post1 = frames(:, :, :, fCnt+1);
            post2 = frames(:, :, :, fCnt-2);
    elseif fCnt == testFramesCnt
            prev2 = frames(:, :, :, fCnt-2);
            prev1 = frames(:, :, :, fCnt-1);
            post1 = frames(:, :, :, fCnt-1);
            post2 = frames(:, :, :, fCnt-2);
    else
            prev2 = frames(:, :, :, fCnt-2);
            prev1 = frames(:, :, :, fCnt-1);
            post1 = frames(:, :, :, fCnt+1);
            post2 = frames(:, :, :, fCnt+2);
    end
    
    prev1 = rgb2gray(prev1);
    curr = rgb2gray(frames(:, :, :, fCnt));
    post1 = rgb2gray(post1);
    
    a = abs(prev1 - curr);
    b = abs(post1 - curr);
    
    Kps(fCnt) = mean(a(:));
    Kns(fCnt) = mean(b(:));
    Alls(fCnt) = (Kps(fCnt) + Kns(fCnt)) / 2;
end

psnr_list = zeros(length(filepaths), 5, 1, 10);

i = 1;
im_inits = deinterlace_video(frames, testFramesCnt);
for iter_index = 10:10
    iter = iter_index * 10;
    [im_dds, im_fusions, running_time] = DeepDeinterlacing(frames, iter*1000);
    
    cnt = size(frames, 4);  
    psnr_inits = zeros(1, cnt);
    psnr_dds = zeros(1, cnt);
    psnr_fusions = zeros(1, cnt);
    for frameCnt = 1:testFramesCnt
        frame = frames(:, :, :, frameCnt);
        im_init = im_inits(:, :, :, frameCnt);
        im_dd = im_dds(:, :, :, frameCnt);
        im_fusion = im_fusions(:, :, :, frameCnt);
        
        psnr_inits(frameCnt) = compute_psnr(frame, im_init);
        psnr_dds(frameCnt) = compute_psnr(frame, im_dd);
        psnr_fusions(frameCnt) = compute_psnr(frame, im_fusion);
        
        continue;
        
        if frameCnt == 2
            fig = findobj('Tag', 'My1stFigure');
            if isempty(fig)
                fig = figure('Tag', 'My1stFigure', 'Name','PSNR Comparison', 'Position', [100, 100, 1049, 895]);
            end
            figure(fig);             
            
            subplot(2, 2, 1), imshow(frame); title('Ground truth');
            subplot(2, 2, 2), imshow(im_init); title([num2str(iter) '000 Init Deinterlace: ' num2str(psnr_inits(frameCnt))]);
            subplot(2, 2, 3), imshow(im_dd); title([num2str(iter) '000 Deep Deinterlace: ' num2str(psnr_dds(frameCnt))]);
            subplot(2, 2, 4), imshow(im_fusion); title([num2str(iter) '000 Deinterlace Fusion: ' num2str(psnr_fusions(frameCnt))]);
        end
    end
    
    psnr_list(i, 1, :, iter_index) = iter;
    psnr_list(i, 2, :, iter_index) = mean(psnr_inits(:));
    psnr_list(i, 3, :, iter_index) = mean(psnr_dds(:));
    psnr_list(i, 4, :, iter_index) = mean(psnr_fusions(:));
end

%plot(1:testFramesCnt, Alls, 'k');
plot(40:testFramesCnt, psnr_dds(40:end), 'k');

%plot(40:testFramesCnt, psnr_dds(40:end), 'k', ...
%     40:testFramesCnt, psnr_inits(40:end), 'r');

%Xi = 40:0.1:testFramesCnt;
%Yi = pchip(40:testFramesCnt,psnr_inits(40:end),Xi);
ylim([30 46])
xlim([40 100])