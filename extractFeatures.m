trainingDir   = 'C:\Users\Anurag\Desktop\mcoc\horse_donkey\TL\Training';
testingDir = 'C:\Users\Anurag\Desktop\mcoc\horse_donkey\TL\Testing';


% imageSet recursively scans the directory tree containing the images.
trainingSet = imageSet(trainingDir,   'recursive');
testSet     = imageSet(testingDir, 'recursive');

img = rgb2gray(read(trainingSet(1), 1));
[hog_16x16, vis16x16] = extractHOGFeatures(img,'CellSize',[64 64]);


cellSize = [64 64];
hogFeatureSize = length(hog_16x16);

%Extract training eatures and labels
trainingFeatures = [];
testingFeatures = [];
trainingLabels   = [];
no_of_classes = numel(trainingSet);
no_of_images = trainingSet(1).Count;

trainSize = 0;
foreground_size = trainingSet(1).Count;
for digit = 1:numel(trainingSet)

    numImages = trainingSet(digit).Count;
    features  = zeros(numImages, hogFeatureSize, 'single');

    for i = 1:numImages
        trainSize = trainSize + 1;
        img = rgb2gray(read(trainingSet(digit), i));

        features(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
    end

    % Mapped objects to numbers 1, 2, 3, 4, 5
    labels = repmat(num2str(digit), numImages, 1);
    
    trainingFeatures = [trainingFeatures; features];   %#ok<AGROW>

end

test_labels = [];

for digit = 1:numel(testSet)

    numImages = testSet(digit).Count;
    features  = zeros(numImages, hogFeatureSize, 'single');

    for i = 1:numImages

        img = rgb2gray(read(testSet(digit), i));

        features(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
    end

   if digit == 1
        labels = ones(1, numImages)';
     else
         labels = -1*ones(1, numImages)';
     end
     
    test_labels = [test_labels; labels];
    
    testingFeatures = [testingFeatures; features];   %#ok<AGROW>

end

test_labels = test_labels';
save('features', 'trainingFeatures','hogFeatureSize', 'foreground_size', 'no_of_images', 'testingFeatures', 'test_labels', 'trainSize');
