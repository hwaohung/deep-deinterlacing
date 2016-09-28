close all; clear all;

folder = 'Test';
iter_max = 110;
input_channels = 3;
testFramesCnt = 30;

filepaths = dir(fullfile(folder, '*.avi'));
psnr_list = zeros(length(filepaths), 5, 1, iter_max);

for iter = 1:iter_max
    for i = 1:length(filepaths)
        clear v frames psnr_gcbis psnr_ds psnr_dsns psnr_fusions;
        
        v = VideoReader(fullfile(folder, filepaths(i).name));
        frameCnt = 0;
        while hasFrame(v)
            frameCnt = frameCnt + 1;
            frame = readFrame(v);
            frames(:, :, frameCnt) = modcrop(frame, 2);
            
            if frameCnt == testFramesCnt
                break;
            end
        end
        
        [im_hs, im_h_dsns, im_h_fusions, running_time] = DeepDeinterlacing(frames, input_channels, iter*1000);
        
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
            
            if frameCnt == 2
                figure(1), imshow(im_b); title([num2str(iter) '000 Deinterlacing Interpolation:' num2str(psnr_gcbi)]);
                figure(2), imshow(im_h); title([num2str(iter) '000 DeepDeinterlacing Reconstruction:' num2str(psnr_d)]);
                figure(3), imshow(im_h_dsn); title([num2str(iter) '000 DeepDeinterlacing Reconstruction(DSN):' num2str(psnr_dsn)]);
                figure(4), imshow(im_h_fusion); title([num2str(iter) '000 DeepDeinterlacing Reconstruction(Fusion):' num2str(psnr_fusion)]);
                %pause(5);
            end
            
            psnr_gcbis(frameCnt) = psnr_gcbi;
            psnr_ds(frameCnt) = psnr_d;
            psnr_dsns(frameCnt) = psnr_dsn;
            psnr_fusions(frameCnt) = psnr_fusion;
        end
        
        psnr_list(i,1,:,iter) = iter;
        psnr_list(i,2,:,iter) = mean(psnr_gcbis(:));
        psnr_list(i,3,:,iter) = mean(psnr_ds(:));
        psnr_list(i,4,:,iter) = mean(psnr_dsns(:));
        psnr_list(i,5,:,iter) = mean(psnr_fusion(:));
        % TODO: Fill other method result
        % EELA
        psnr_list(i,6,:,iter) = 33.242;
        % NEDD
        psnr_list(i,7,:,iter) = 34.135;
        % RF-BF
        psnr_list(i,8,:,iter) = 35.764;
    end
end

avg_psnr_list(1,:) = [1:iter_max];
for iter = 1:iter_max
    avg_psnr_list(2,iter) = mean(psnr_list(:,3,:,iter));
    avg_psnr_list(3,iter) = mean(psnr_list(:,4,:,iter));
    avg_psnr_list(4,iter) = mean(psnr_list(:,5,:,iter));
end

%% Display iteration PSNR result
plot(avg_psnr_list(1,:), avg_psnr_list(2,:), 'r',...
     avg_psnr_list(1,:), avg_psnr_list(3,:), 'g',...
     avg_psnr_list(1,:), avg_psnr_list(4,:), 'b');

title('Deep Deinterlacing');
xlabel('Iteration');
ylabel('PSNR');
legend('DeepDeinterlacing', 'DeepDeinterlacing(DSN)', 'DeepDeinterlacing(Fusion)');
