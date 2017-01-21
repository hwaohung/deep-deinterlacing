clear;
close all;

iter_max = 100;
iter_step = 10;
folder = 'Test';
testFramesCnt = 300;

filepaths = dir(fullfile(folder, '*.avi'));
psnr_list = zeros(length(filepaths), 5, 1, floor(iter_max/iter_step));

filepaths = filepaths(8);

for i = 1:length(filepaths)
    clear frames psnr_inits psnr_dds psnr_fusions; 
    frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
    im_inits = deinterlace_video(frames, testFramesCnt);
    
    for iter_index = 10:floor(iter_max/iter_step)
        iter = iter_index * iter_step;
        [im_dds, im_fusions, running_time] = DeepDeinterlacing(frames, iter*1000);
        
        cnt = size(frames, 4);  
        psnr_inits = zeros(1, cnt);
        psnr_dds = zeros(1, cnt);
        psnr_fusions = zeros(1, cnt);
        for frameCnt = 1:cnt
            frame = frames(:, :, :, frameCnt);
            im_init = im_inits(:, :, :, frameCnt);
            im_dd = im_dds(:, :, :, frameCnt);
            im_fusion = im_fusions(:, :, :, frameCnt);
            
            psnr_inits(frameCnt) = compute_psnr(frame, im_init);
            psnr_dds(frameCnt) = compute_psnr(frame, im_dd);
            psnr_fusions(frameCnt) = compute_psnr(frame, im_fusion);
            
            result(:, :, :, frameCnt) = im_dd;
            
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
end

for iter_index = 1:floor(iter_max/iter_step)
    iter = iter_index * iter_step;
    avg_psnr_list(1, iter_index) = iter;
    avg_psnr_list(2, iter_index) = mean(psnr_list(:, 2, :, iter_index));
    avg_psnr_list(3, iter_index) = mean(psnr_list(:, 3, :, iter_index));
    avg_psnr_list(4, iter_index) = mean(psnr_list(:, 4, :, iter_index));
end

%% Display iteration PSNR result
plot(avg_psnr_list(1, :), avg_psnr_list(2, :), 'k',...
     avg_psnr_list(1, :), avg_psnr_list(3, :), 'r',...
     avg_psnr_list(1, :), avg_psnr_list(4, :), 'g');

title('PSNR Comparison');
xlabel('Iteration');
ylabel('PSNR');
legend('Init Deinterlace', 'Deep Deinterlace', 'Deinterlace Fusion');
