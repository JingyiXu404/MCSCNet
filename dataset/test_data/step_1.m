clear;close all;

%% initialization
folder_gt='xxx'; % path of '.png'
count = 0;
noiseSigma=70;
%% generate data
filepaths_gt=[]; 
filepaths_gt = [filepaths_gt; dir(fullfile(folder_gt, '*.png'))]; 

for i = 1:68
                                                   
                    image = im2double(imread(fullfile(folder_gt,filepaths_gt(i).name)));                    
                    image_noise = single(image + noiseSigma/255*randn(size(image)));
                  %% save images
                    if i<10 
                        imwrite(image_noise,strcat("gaussian_testset/noise",num2str(noiseSigma),"_BSD68/00",num2str(i),".png"));
                    elseif i<100
                        imwrite(image_noise,strcat("gaussian_testset/noise",num2str(noiseSigma),"_BSD68/0",num2str(i),".png"));
                    else 
                        imwrite(image_noise,strcat("gaussian_testset/noise",num2str(noiseSigma),"_BSD68/",num2str(i),".png"));
                    end
                  %% save mat
                    [hei,wid,~] = size(image);
                    filepaths_gt(i).name
                    count=count+1;
                    %initialization
                    BSD68_test_noise70 = zeros(hei, wid, 3, 1);
                    BSD68_test_label70 = zeros(hei, wid, 3, 1);
                    %write
                    BSD68_test_noise70(:, :, :, 1) = image_noise;
                    BSD68_test_label70(:, :, :, 1) = image;
                    BSD68_test_noise70(:, :, :, 2) = image_noise;
                    BSD68_test_label70(:, :, :, 2) = image;
                    clear image image_noise 
                    %save mat
                    if exist('gaussian_testset/MAT/BSD68_test_noise70/')==0
                        mkdir('gaussian_testset/MAT/BSD68_test_noise70/');
                    end
                    name_label=['gaussian_testset/MAT/BSD68_test_noise70/BSD68_test_label_',num2str(count),'.mat'];
                    save(name_label,'BSD68_test_label70','-v7.3');
                    clear BSD68_test_label70
                    name_LR=['gaussian_testset/MAT/BSD68_test_noise70/BSD68_test_noise70_',num2str(count),'.mat'];
                    save(name_LR,'BSD68_test_noise70','-v7.3');
                    clear BSD68_test_noise70
                              
end


