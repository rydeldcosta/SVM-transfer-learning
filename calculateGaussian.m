function [features_gauss] = calculateGaussian(index)
    global features;
    global landmarks;
    sigma = 0.5;
    
    for i=1:size(landmarks,1)
        s=norm(landmarks(i,:)-features(index,:));
        features_gauss(i)=exp(-s*s/(2*sigma*sigma));
    end


    
