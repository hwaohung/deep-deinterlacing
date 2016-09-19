close all; clear all;

folder = 'Test';
iter_max = 110;
use_gpu = 0;

filepaths = dir(fullfile(folder, '*.avi'));
psnr_list = zeros(length(filepaths), 5, 1, iter_max);

for iter = 1:iter_max
    for i = 1:length(filepaths)
        v = VideoReader(fullfile(folder, filepaths(i).name));
        
        frameCnt = 0;
        clear psnr_gcbis psnr_ds psnr_dsns psnr_fusions;
        while hasFrame(v)
            frameCnt = frameCnt + 1;
            im_gnd = readFrame(v);            
            im_gnd = modcrop(im_gnd, 2);
            
            [im_b] = interlace(im_gnd, mod(frameCnt, 2));
            [im_h, im_h_dsn, im_h_fusion, running_time] = DeepDeinterlacing(im_gnd, mod(frameCnt, 2), use_gpu, iter*1000);
            
            psnr_gcbi = compute_psnr(im_gnd(:, :, 1), im_b(:, :, 1));
            psnr_d = compute_psnr(im_gnd(:, :, 1), im_h(:, :, 1));
            psnr_dsn = compute_psnr(im_gnd(:, :,1), im_h_dsn(:, :, 1));
            psnr_fusion = compute_psnr(im_gnd(:, :,1), im_h_fusion(:, :, 1));
            
            figure(1), imshow(im_b); title(['Deinterlacing Interpolation:' num2str(psnr_gcbi)]);
            figure(2), imshow(im_h); title([num2str(iter) '000 DeepDeinterlacing Reconstruction:' num2str(psnr_d)]);
            figure(3), imshow(im_h_dsn); title([num2str(iter) '000 DeepDeinterlacing Reconstruction(DSN):' num2str(psnr_dsn)]);
            figure(3), imshow(im_h_fusion); title([num2str(iter) '000 DeepDeinterlacing Reconstruction(Fusion):' num2str(psnr_fusion)]);
            pause(1);
            
            psnr_gcbis(frameCnt) = psnr_gcbi;
            psnr_ds(frameCnt) = psnr_d;
            psnr_dsns(frameCnt) = psnr_dsn;
            psnr_fusions(frameCnt) = psnr_fusion;
        end
        
        psnr_list(i,1,:,iter) = iter;
        psnr_list(i,2,:,iter) = mean(psnr_gcbis(:));
        psnr_list(i,3,:,iter) = mean(psnr_ds(:));
        psnr_list(i,4,:,iter) = mean(psnr_dsns(:));
        % TODO: Fill other method result
        psnr_list(i,5,:,iter) = mean(psnr_fusion(:));
    end
end

avg_psnr_list(1,:) = [1:iter_max];
for iter = 1:iter_max
    avg_psnr_list(2,iter) = mean(psnr_list(:,3,:,iter));
    avg_psnr_list(3,iter) = mean(psnr_list(:,4,:,iter));
    avg_psnr_list(4,iter) = mean(psnr_list(:,5,:,iter));
end