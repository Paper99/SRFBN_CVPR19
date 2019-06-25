function Prepare_TrainData_HR_LR()
clear; close all; clc

% SET scale factor here
scale = 4;

% whether use off-line data augmentation
USE_AUG = true;

% SET data dir
sourcedir = '/home/ser606/Downloads/DIV2K_train_HR';
savedir = '/media/ser606/Data/paper99/DIV2K/Augment';

saveHRpath = fullfile(savedir, 'DIV2K_train_HR_aug', ['x' num2str(scale)]);
saveLRpath = fullfile(savedir, 'DIV2K_train_LR_aug', ['x' num2str(scale)]);

if ~exist(saveHRpath, 'dir')
    mkdir(saveHRpath);
end
if ~exist(saveLRpath, 'dir')
    mkdir(saveLRpath);
end

filepaths = [dir(fullfile(sourcedir, '*.png'));dir(fullfile(sourcedir, '*.bmp'))];

downsizes = [0.8, 0.7, 0.6, 0.5];

% prepare data with or without augmentation
parfor i = 1 : length(filepaths)
    filename = filepaths(i).name;
    fprintf('No.%d -- Processing %s\n', i, filename);
    [add, im_name, type] = fileparts(filepaths(i).name);
    image = imread(fullfile(sourcedir, filename));
    if USE_AUG
        for angle = 0 : 2 : 2
            for downidx = 0 : 1 : length(downsizes)
                image_HR = image;
                if downidx > 0
                    image_HR = imresize(image_HR, downsizes(downidx), 'bicubic');
                end
                image_HR = rot90(image_HR, angle);
                image_HR = modcrop(image_HR, scale);
                image_LR = imresize(image_HR, 1/scale, 'bicubic');

                saveHRfile =  [im_name '_rot' num2str(angle*90) '_ds' num2str(downidx) '.png'];
                saveLRfile = [im_name '_rot' num2str(angle*90) '_ds' num2str(downidx) '.png'];

                imwrite(image_HR, fullfile(saveHRpath, saveHRfile));    
                imwrite(image_LR, fullfile(saveLRpath, saveLRfile));               
            end          
        end
    else
        image_HR = modcrop(image, scale);
        image_LR = imresize(image_HR, 1/scale, 'bicubic');
        
        imwrite(image_HR, fullfile(saveHRpath, [im_name '.png']));    
        imwrite(image_LR, fullfile(saveLRpath, [im_name '.png']));
    end
end

end

function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end
