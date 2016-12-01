function [loss] = hinge_loss(passed_parameters, index)
  
    global labels;
    global features;
    
    [farjee_len len] = size(features);
    temp1 = dot(passed_parameters,features(index, 1:len));
    temp = 1 - temp1*labels(index);
    if temp<0
        temp = 0;
    end
    loss = temp;