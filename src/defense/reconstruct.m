function [img_rec] = reconstruct(img_set)
  
    I_ref = reshape(img_set(1,:,:,:),[size(img_set,2),size(img_set,3),size(img_set,4)]);
    
    img_rec = zeros(size(I_ref));
    
    for c=1:3
        % recombine step
        for i=2:size(img_set,1)
            I1 = I_ref(:,:,c);
            I2 = reshape(img_set(i,:,:,c),size(I1));
            alpha = ones(size(I1));
            l1_norm =  norm(I1-I2,1); % norm(I2/255-I1/255,1) when square attack used to make the copy 
            alpha(I1~=I2) = 1/l1_norm;
            img_rec(:,:,c) = (alpha).*I1 + (1-alpha).*I2;
        end
    end
    img_rec = double(uint8(img_rec));
end