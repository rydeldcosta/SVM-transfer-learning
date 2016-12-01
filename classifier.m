accuracy = [];
global tau;

%for m=0.5:0.5:10
data = load('features.mat');
global trainSize
trainSize = data.trainSize;

global foreground_size
foreground_size = data.foreground_size;

source = load('source_params.mat');
global source_parameters
source_parameters = source.final_parameters;

trainingFeatures = data.trainingFeatures;

hogFeatureSize = data.hogFeatureSize;
tau = 6;
global features;
global landmarks;
global parameters;
global labels;
global lambda;
lambda = 4;

features = data.trainingFeatures;
landmarks = data.trainingFeatures;  % these are constant

labels = [ones(1, foreground_size) -1*ones(1, trainSize - foreground_size)];

initial_parameters = zeros(1,size(features,2));


options=optimoptions('fminunc', 'Algorithm','quasi-newton','GradObj','on','Display','iter', 'MaxIter', 70);
[final_parameters,fval,exitflag,output] = fminunc(@cost_function_parameters, initial_parameters,options);

parameters = final_parameters;


save('parameters.mat' , 'final_parameters');

acc = testing()
%accuracy = [acc; accuracy]



