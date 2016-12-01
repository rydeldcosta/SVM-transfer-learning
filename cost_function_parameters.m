function [min, gradient] = cost_function_parameters(temp_parameters)
 global features
 global labels
 global source_parameters
 
 data = load('features.mat');
 trainSize = data.trainSize;
 global lambda;
 global tau;
 
regularization = norm(temp_parameters - source_parameters*tau,2)*norm(temp_parameters - source_parameters*tau,2);
 %regularization = norm(temp_parameters,1)*norm(temp_parameters,1);
 
 
 
 % loop runs for number of training examples in total
 % in main, convert all features to gaussian initially and assign to global
 % variable features
 
min3=0;
 for i=1:trainSize
    min3 = min3 + lambda*hinge_loss(temp_parameters, i);
 end
 
 min = double(regularization + min3);
 

 len = size(temp_parameters,2);

gradient = zeros(1, len);
for i=1:len
    for j=1:trainSize
        if dot(temp_parameters,features(j,:))*labels(j)<1
            gradient(i)= gradient(i) +  lambda*(-features(j,i)*labels(j));
        end
    end
    gradient(i)= gradient(i) +  2*(temp_parameters(i)-source_parameters(i));
%  gradient(i)= gradient(i) +  2*temp_parameters(i);
end


