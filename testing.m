function [accuracy] = testing()
data = load('features.mat');
parameters = load('parameters.mat');

testFeatures = data.testingFeatures;

final_parameters = parameters.final_parameters;

test_labels = data.test_labels;

global features;
features = testFeatures;
test_size = size(testFeatures, 1);

% for i=1:test_size
%    testFeaturesGauss(i, :) = calculateGaussian(i); % RK Changed 
% end
% features = testFeaturesGauss; % RK CHANGED 

accuracy = sum(sign(final_parameters*features')==test_labels)/test_size;

end