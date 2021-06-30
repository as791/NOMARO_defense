# Copyright (C) 2021, Aryaman Sinha
clc;
clear;
close all;
%% initialize num of extra copies and classifier
n = 1;
% select the classifier
% net = vgg16;
% net = inceptionv3;
net = resnet101;
sigma = 0.02;  
std = sqrt(3);
addpath('bm3d');
%% read dir and extract filenames of the data 
adv_copies_dir = '../test_bed/multi_adv_database/cw_attack/adv/resnet/cw_l2/';
orig_copies_dir = '../test_bed/multi_adv_database/cw_attack/orig/resnet/';
adv_database_dir = '../test_bed/testing_database/resnet/cw_l2/';
orig_database_dir = '../test_bed/testing_database/resnet/test_set/';

files = dir(orig_database_dir);
directoryNames = {files.name};
orig_input = containers.Map('KeyType','uint32','ValueType','any');
for f = directoryNames
    f1 = f{1};
    if contains(f1,'_')
        data = strsplit(f1,'_');
        if contains(f1,'orig')
            orig_input(str2double(data(2))+1)=f1;
        end
    end
end

files = dir(adv_database_dir);
directoryNames = {files.name};
adv_input = containers.Map('KeyType','uint32','ValueType','any');
for f = directoryNames
    f1 = f{1};
    if contains(f1,'_')
        data = strsplit(f1,'_');
        if contains(f1,'adv')
            adv_input(str2double(data(2))+1)=f1;
        end
    end
end

files = dir(adv_copies_dir);
directoryNames = {files.name};
adv_copies = containers.Map('KeyType','char','ValueType','any');
for f=directoryNames
    f1 = f{1};
    if contains(f1,'_')
        data = strsplit(f1,'_');
        if contains(f1,'.mat')&& str2double(data(3))<=n
            adv_copies(strcat(data{2},'$',data{3}))=f1;
        end
    end
end

files = dir(orig_copies_dir);
directoryNames = {files.name};
orig_copies = containers.Map('KeyType','char','ValueType','any');
for f=directoryNames
    f1 = f{1};
    if contains(f1,'_')
        data = strsplit(f1,'_');
        if contains(f1,'.mat')&& str2double(data(3))<=n 
            orig_copies(strcat(data{2},'$',data{3}))=f1;
        end
    end
end
%% testing for adv input
k = n+1;
cnt=0;
num = 100;
sc=[];
for sigma = linspace(0,1,100)
for i=1:num
    % load image from .mat files
    if isKey(adv_input,i)
        file1 = adv_input(i);
        file2 = orig_input(i);
        test_img = double(load(strcat(adv_database_dir,file1)).adv)*255;
        orig_img = double(load(strcat(orig_database_dir,file2)).orig)*255;
        % image set contianing all the copies we have 
        img_set = zeros(k,size(test_img,1),size(test_img,2),size(test_img,3));
        img_set(1,:,:,:)=test_img;
        for j=1:k-1
            data = strsplit(file1,'_');
            file = adv_copies(strcat(data{2},'$',num2str(j)));
            copy_img = double(load(strcat(adv_copies_dir,file)).multi)*255;
            img_set(j+1,:,:,:)=copy_img;
        end
        img = reconstruct(img_set);
		
		% applying image denoising methods
        img = wavelet_denoising(img,sigma);  
%         img = CBM3D(img,std);

        if classify(net,img) == classify(net,orig_img)
            cnt = cnt+1;
        end
    end
end
sc(end+1,:)=cnt;
end
fprintf('Success Rate: %.4f\n',cnt*100/size(adv_input,1));
%% testing for orig input
k = n+1;
cnt = 0;
num = size(orig_input,1); 
for i=1:num
    % load image from .mat file
    file1 = orig_input(i);
    orig_img = double(load(strcat(orig_database_dir,file1)).orig)*255;

    % making image set of all copies we have 
    img_set = zeros(k,size(orig_img,1),size(orig_img,2),size(orig_img,3));
    img_set(1,:,:,:)=orig_img;
    for j=1:k-1
        data = strsplit(file1,'_');
        file = orig_copies(strcat(data{2},'$',num2str(j)));
        copy_img = double(load(strcat(orig_copies_dir,file)).multi)*255;
        img_set(j+1,:,:,:)=copy_img;
    end
    img = reconstruct(img_set);
	
	% applying image denoising methods
    img = wavelet_denoising(img,sigma);
%     img = CBM3D(img,std);

    PSNR = psnr(img,orig_img,255);
    
    if classify(net,img) == classify(net,orig_img)
        cnt = cnt+1;
    end
end
fprintf('Accuracy: %.4f\n',cnt*100/size(orig_input,1));
