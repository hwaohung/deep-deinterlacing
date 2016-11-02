clear;
close all;

iter_max = 100;
iter_step = 10;
folder = 'Test';
testFramesCnt = 30;

filepaths = dir(fullfile(folder, '*.avi'));
psnr_list = zeros(length(filepaths), 5, 1, floor(iter_max/iter_step));

for i = 1:length(filepaths)
    clear frames psnr_gcbis psnr_ds psnr_dsns psnr_fusions; 
    frames = get_video_frames(fullfile(folder, filepaths(i).name), testFramesCnt);
    %[resizeds1, resizeds2] = interlaced_resize(frames);
    %frames = cat(4, resizeds1, resizeds2);
    
    for iter_index = 10:floor(iter_max/iter_step)
        iter = iter_index * iter_step;
        [im_hs, im_h_dsns, im_h_fusions, running_time] = DeepDeinterlacing(frames, iter*1000);
        
        psnr_gcbis = size(1, size(frames, 3));
        psnr_d = size(1, size(frames, 3));
        psnr_dsns = size(1, size(frames, 3));
        psnr_fusions = size(1, size(frames, 3));
        for frameCnt = 1:size(frames, 3)
            frame = frames(:, :, frameCnt);
            [im_b] = interlace(frame, mod(frameCnt, 2));
            im_b = deinterlace(im_b, mod(frameCnt, 2));
            im_h = im_hs(:, :, frameCnt);
            im_h_dsn = im_h_dsns(:, :, frameCnt);
            im_h_fusion = im_h_fusions(:, :, frameCnt);
                                   
            psnr_gcbi = compute_psnr(frame, im_b);
            psnr_d = compute_psnr(frame, im_h);
            psnr_dsn = compute_psnr(frame, im_h_dsn);
            psnr_fusion = compute_psnr(frame, im_h_fusion);
            
            psnr_gcbis(frameCnt) = psnr_gcbi;
            psnr_ds(frameCnt) = psnr_d;
            psnr_dsns(frameCnt) = psnr_dsn;
            psnr_fusions(frameCnt) = psnr_fusion;
            
            if frameCnt == 1
                fig = findobj('Tag', 'My1stFigure');
                if isempty(fig)
                    fig = figure('Tag', 'My1stFigure', 'Name','PSNR Comparison', 'Position', [100, 100, 1049, 895]);
                end
                figure(fig);             
                
                subplot(2, 2, 1), imshow(frame); title('Ground truth');
                subplot(2, 2, 2), imshow(im_b); title([num2str(iter) '000 Linear Interpolation: ' num2str(psnr_gcbi)]);
                subplot(2, 2, 3), imshow(im_h); title([num2str(iter) '000 DeepDeinterlacing: ' num2str(psnr_d)]);
                subplot(2, 2, 4), imshow(im_h_dsn); title([num2str(iter) '000 DeepDeinterlacing(DSN): ' num2str(psnr_dsn)]);
                pause(0.1);
            end
        end
        
        v = VideoWriter(num2str(i), 'Grayscale AVI');
        open(v);
    
        [rowCount, colCount, frameCount] = size(im_hs);
        for tt = 1:frameCount
            writeVideo(v, im_hs(:,:,tt));
        end
    
        close(v);
        
        t1(i, :) = psnr_gcbis(:);
        t2(i, :) = psnr_ds(:);
        
        psnr_list(i,1,:,iter_index) = iter;
        psnr_list(i,2,:,iter_index) = mean(psnr_gcbis(:));
        psnr_list(i,3,:,iter_index) = mean(psnr_ds(:));
        psnr_list(i,4,:,iter_index) = mean(psnr_dsns(:));
        psnr_list(i,5,:,iter_index) = mean(psnr_fusion(:));
    end
end

for iter_index = 1:floor(iter_max/iter_step)
    iter = iter_index * iter_step;
    avg_psnr_list(1, iter_index) = iter;
    avg_psnr_list(2, iter_index) = mean(psnr_list(:, 2, :, iter_index));
    avg_psnr_list(3, iter_index) = mean(psnr_list(:, 3, :, iter_index));
    avg_psnr_list(4, iter_index) = mean(psnr_list(:, 4, :, iter_index));
    avg_psnr_list(5, iter_index) = mean(psnr_list(:, 5, :, iter_index));
end

%% Display iteration PSNR result
plot(avg_psnr_list(1, :), avg_psnr_list(2, :), 'k',...
     avg_psnr_list(1, :), avg_psnr_list(3, :), 'r',...
     avg_psnr_list(1, :), avg_psnr_list(4, :), 'g',...
     avg_psnr_list(1, :), avg_psnr_list(5, :), 'b');

title('Deep Deinterlacing');
xlabel('Iteration');
ylabel('PSNR');
legend('Linear', 'DeepDeinterlacing', 'DeepDeinterlacing(DSN)', 'DeepDeinterlacing(Fusion)');
