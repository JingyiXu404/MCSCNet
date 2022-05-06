% 
clear;close all;
folder_gt='train';
size_input = 64;
size_label = 64;
stride = 31;

%% initialization
data_noise_50_01 = zeros(size_input, size_input, 3, 1);
data_label_50_01 = zeros(size_label, size_label, 3, 1);
noiseSigma=50; % noise level

count = 0;

%% generate data
filepaths_gt=[]; 
filepaths_gt = [filepaths_gt; dir(fullfile(folder_gt, '*.jpg'))]; 

%% make mat
for i = 1:432
                       
                        image = im2double(imread(fullfile(folder_gt,filepaths_gt(i).name)));                    
                        image_noise = single(image + noiseSigma/255*randn(size(image)));
                        [hei,wid,~] = size(image);
%                         filepaths_gt(i).name
                        for x = 1 : stride : hei-size_input+1
                            for y = 1 :stride : wid-size_input+1
                                 subim_inputx = image_noise(x : x+size_input-1, y : y+size_input-1,:);
                                subim_label = image(x : x+size_label-1, y : y+size_label-1,:); 

                                count=count+1;
                                data_noise_50_01(:, :, :, count) = subim_inputx;
                                data_label_50_01(:, :, :, count) = subim_label;
                            end
                        end
                         clear image
                         clear image_noise subim_inputx subim_label
end

order = randperm(count);
data_noise_50_01 = data_noise_50_01(:, :, :, order);
save('data_noise_50_01.mat','data_noise_50_01','-v7.3');
clear data_noise_50_01
data_label_50_01 = data_label_50_01(:, :, :, order); 
save('data_label_50_01.mat','data_label_50_01','-v7.3');
clear data_label_50_01


