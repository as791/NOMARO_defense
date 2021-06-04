function [img] = wavelet_denoising(img,sigma)
    img = double(py.skimage.restoration.denoise_wavelet(pyargs('image',py.numpy.array(img/255), 'sigma',sigma(1), 'mode','soft', 'multichannel',true,... 
    'convert2ycbcr',true,'method','BayesShrink','wavelet','db1')))*255;
end