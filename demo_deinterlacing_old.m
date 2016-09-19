% =========================================================================
% Test code for Super-Resolution Convolutional Neural Networks (SRCNN)
%
% Reference
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, 
%   in Proceedings of European Conference on Computer Vision (ECCV), 2014
%
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks,
%   arXiv:1501.00092
%
% Chao Dong
% IE Department, The Chinese University of Hong Kong
% For any question, send email to ndc.forward@gmail.com
% =========================================================================

close all;
clear all;

folder = 'Test';
iter_max = 110;
filepaths = dir(fullfile(folder,'*.avi'));
%psnr_list = zeros(2,length(filepaths));
%psnr_list = zeros(5,3,iter_max);
% 5: iter and method amounts
% 3: rgb
psnr_list = zeros(length(filepaths),5,3,iter_max);
cpsnr_list = zeros(length(filepaths),3,iter_max);
for i = 1 : length(filepaths)
for iter = 1:iter_max
%for i = 4 : 4
    %% read ground truth image
    im  = imread(fullfile(folder,filepaths(i).name));
    %im  = imread('Set5\butterfly_GT.bmp');
    %im  = imread('Set14\zebra.bmp');

    %% set parameters
    up_scale = 2;
    %model = 'model\9-5-5(ImageNet)\x3.mat';
    % up_scale = 3;
    % model = 'model\9-3-5(ImageNet)\x3.mat';
    % up_scale = 3;
    % model = 'model\9-1-5(91 images)\x3.mat';
    % up_scale = 2;
    % model = 'model\9-5-5(ImageNet)\x2.mat'; 
    % up_scale = 4;
    % model = 'model\9-5-5(ImageNet)\x4.mat';

    %% work on illuminance only
    
    im_gnd = modcrop(im, up_scale);
    im_gnd = single(im_gnd)/255;

    %% bicubic interpolation
    [imss im_l] = mosaic(im_gnd, 'gbrg');
    im_b = im2double(demosaic(im2uint8(imss),'gbrg'));

    %% SRCNN
    %im_h = SRCNN(model, im_b);
    use_gpu = 0;
    [im_h running_time im_h_dsn] = DeepDemosaicing(im_l, use_gpu, iter*1000);

    %% remove border
    %im_h = shave(uint8(im_h * 255), [up_scale, up_scale]);
    im_b = uint8(im_b * 255);
    im_h = uint8(im_h * 255);
    im_h_dsn = uint8(im_h_dsn * 255);
    im_gnd = uint8(im_gnd * 255);

    %% compute PSNR
    psnr_gcbi1 = compute_psnr(im_gnd(:,:,1),im_b(:,:,1));
    psnr_gcbi2 = compute_psnr(im_gnd(:,:,2),im_b(:,:,2));
    psnr_gcbi3 = compute_psnr(im_gnd(:,:,3),im_b(:,:,3));
    psnr_d1 = compute_psnr(im_gnd(:,:,1),im_h(:,:,1));
    psnr_d2 = compute_psnr(im_gnd(:,:,2),im_h(:,:,2));
    psnr_d3 = compute_psnr(im_gnd(:,:,3),im_h(:,:,3));
    psnr_dsn1 = compute_psnr(im_gnd(:,:,1),im_h_dsn(:,:,1));
    psnr_dsn2 = compute_psnr(im_gnd(:,:,2),im_h_dsn(:,:,2));
    psnr_dsn3 = compute_psnr(im_gnd(:,:,3),im_h_dsn(:,:,3));

    %% show results
    %fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
    %fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);
    %fprintf('PSNR for SRCNN dsn2 Reconstruction: %f dB\n', compute_psnr(im_gnd(3:end-2,3:end-2),im_h_dsn2));

    figure(1), imshow(im_b); title(['Demosaicking Interpolation R:' num2str(psnr_gcbi1) ' G:' num2str(psnr_gcbi2) ' B:' num2str(psnr_gcbi3)]);
    figure(i+1), imshow(im_h); title([num2str(iter) '000 DeepDemosaicking Reconstruction R:' num2str(psnr_d1) ' G:' num2str(psnr_d2) ' B:' num2str(psnr_d3) '  Dsn R:' num2str(psnr_dsn1) ' G:' num2str(psnr_dsn2) ' B:' num2str(psnr_dsn3)]);
    pause(1);
    psnr_list(i,1,:,iter) = iter;
    psnr_list(i,2,1,iter) = psnr_gcbi1;
    psnr_list(i,2,2,iter) = psnr_gcbi2;
    psnr_list(i,2,3,iter) = psnr_gcbi3;
    psnr_list(i,3,1,iter) = psnr_dsn1;
    psnr_list(i,3,2,iter) = psnr_dsn2;
    psnr_list(i,3,3,iter) = psnr_dsn3;
    psnr_list(i,4,1,iter) = psnr_d1;
    psnr_list(i,4,2,iter) = psnr_d2;
    psnr_list(i,4,3,iter) = psnr_d3;
    psnr_list(i,5,1,iter) = 41.68;
    psnr_list(i,5,2,iter) = 43.63;
    psnr_list(i,5,3,iter) = 42.51;
    cpsnr_list(i,1,iter) = iter;
    cpsnr_list(i,2,iter) = mean([psnr_d1 psnr_d2 psnr_d3]);
    cpsnr_list(i,3,iter) = mean([psnr_dsn1 psnr_dsn2 psnr_dsn3]);
end
%fprintf('Average PSNR for Demosaicking Interpolation: %f dB\n', mean(psnr_list(1,:)));
%fprintf('iter %d Average PSNR for DeepDemosaicking Reconstruction: %f dB\n', iter, mean(psnr_list(2,:)));
% red_list(:,:) = psnr_list(i,:,1,:);
% green_list(:,:) = psnr_list(i,:,2,:);
% blue_list(:,:) = psnr_list(i,:,3,:);
% figure,plot(red_list(1,:),red_list(2,:),red_list(1,:),red_list(3,:),red_list(1,:),red_list(4,:),red_list(1,:),red_list(5,:)),title(['PSNR in Red(kodak' num2str(i) ')']),xlabel('iterate(*1000)'),ylabel('PSNR'),legend('GCBI','DeepDemosaicking(DSN)','DeepDemosaicking','JD','Location','southeast')
% figure,plot(green_list(1,:),green_list(2,:),green_list(1,:),green_list(3,:),green_list(1,:),green_list(4,:),green_list(1,:),green_list(5,:)),title(['PSNR in Green(kodak' num2str(i) ')']),xlabel('iterate(*1000)'),ylabel('PSNR'),legend('GCBI','DeepDemosaicking(DSN)','DeepDemosaicking','JD','Location','southeast')
% figure,plot(blue_list(1,:),blue_list(2,:),blue_list(1,:),blue_list(3,:),blue_list(1,:),blue_list(4,:),blue_list(1,:),blue_list(5,:)),title(['PSNR in Blue(kodak' num2str(i) ')']),xlabel('iterate(*1000)'),ylabel('PSNR'),legend('GCBI','DeepDemosaicking(DSN)','DeepDemosaicking','JD','Location','southeast')

end

all_cpsnr_list(1,:) = [1:iter_max];
for j = 1:iter_max
    all_cpsnr_list(2,j) = mean(cpsnr_list(:,2,j)); 
    all_cpsnr_list(3,j) = mean(cpsnr_list(:,3,j)); 
end
%figure,plot(all_cpsnr_list(1,:),all_cpsnr_list(2,:),all_cpsnr_list(1,:),all_cpsnr_list(3,:),[1 iter_max],[max(all_cpsnr_list(2,:)) max(all_cpsnr_list(2,:))],[1 iter_max],[max(all_cpsnr_list(3,:)) max(all_cpsnr_list(3,:))]),title(['PSNR in Average']),xlabel('iterate(*1000)'),ylabel('PSNR'),legend('DeepDemosaicking','DeepDemosaicking(DSN)','Location','southeast')

%imwrite(im_b, ['Bicubic Interpolation' '.bmp']);
%imwrite(im_h, ['SRCNN Reconstruction' '.bmp']);
% red_list(:,:) = psnr_list(:,1,:);
% green_list(:,:) = psnr_list(:,2,:);
% blue_list(:,:) = psnr_list(:,3,:);
% figure,plot(red_list(1,:),red_list(2,:),red_list(1,:),red_list(3,:),red_list(1,:),red_list(4,:),red_list(1,:),red_list(5,:)),title('PSNR in Red'),xlabel('iterate(*1000)'),ylabel('PSNR'),legend('GCBI','DeepDemosaicking(DSN)','DeepDemosaicking','JD','Location','southeast')
% figure,plot(green_list(1,:),green_list(2,:),green_list(1,:),green_list(3,:),green_list(1,:),green_list(4,:),green_list(1,:),green_list(5,:)),title('PSNR in Green'),xlabel('iterate(*1000)'),ylabel('PSNR'),legend('GCBI','DeepDemosaicking(DSN)','DeepDemosaicking','JD','Location','southeast')
% figure,plot(blue_list(1,:),blue_list(2,:),blue_list(1,:),blue_list(3,:),blue_list(1,:),blue_list(4,:),blue_list(1,:),blue_list(5,:)),title('PSNR in Blue'),xlabel('iterate(*1000)'),ylabel('PSNR'),legend('GCBI','DeepDemosaicking(DSN)','DeepDemosaicking','JD','Location','southeast')
