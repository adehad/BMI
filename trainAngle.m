clear all; close all;
%%  Loads data
load('monkeydata_training.mat');
tBeginTarget = 1; tEndTarget = 300;

[clusterCentre, meanXTrain] = trainAngleClassification(trial, tBeginTarget, tEndTarget);
[predAngle] = testAngleClassification(trial, tBeginTarget, tEndTarget, clusterCentre, meanXTrain);


function [clusterCentre, meanXTrain] = trainAngleClassification(trials, tBeginTarget, tEndTarget)
    [metricPreMovement labels] = formatPreMovement(trials, tBeginTarget, tEndTarget);
    [preCrtTrain, meanXTrain] = centerData(metricPreMovement);
    numTrials = size(trials,1);
    clusterCentre = extractClusters(preCrtTrain, numTrials);
    
    function [crtTrain, meanXTrain] = centerData(train)
        meanXTrain = mean(train,1);
        crtTrain = train - meanXTrain;
    end
    function [outputTrainMetric, outputLabel] = formatPreMovement(trials, tLow, tHigh)
        numTrials = size(trials,1);
        numAngle = size(trials,2);
        numNeuron = size(trials(1,1).spikes,1);
        numMetrics = 2;
        %tDiff = tHigh - tLow; 
        N = numTrials*numAngle;
        %D = tDiff*numNeuron;

        outputTrainMetric = zeros(numTrials, numAngle, numMetrics*numNeuron);
        outputLabel = zeros(numTrials, numAngle);

        for trial = 1:numTrials
            for angle = 1:numAngle
                metricStore = zeros(numNeuron, numMetrics);
                for neuron = 1:numNeuron
                    lenSpike = length(trials(trial,angle).spikes(neuron,:));
                    if lenSpike < tHigh-1
                        metricStore(neuron,:) = [mean(trials(trial,angle).spikes(neuron , tLow:lenSpike)), std(trials(trial,angle).spikes(neuron , tLow:lenSpike))];
                    else
                        metricStore(neuron,:) = [mean(trials(trial,angle).spikes(neuron , tLow:tHigh-1)), std(trials(trial,angle).spikes(neuron , tLow:tHigh-1))];
                    end                
                end
                outputTrainMetric(trial, angle, :) = metricStore(:);
                outputLabel(trial, angle) = angle;
            end
        end
        outputTrainMetric = reshape(outputTrainMetric, [N numMetrics*numNeuron]);
        outputLabel = reshape(outputLabel, [N 1]);
    end
    function clusterCentre = extractClusters(xTrainNorm, numTrials)
    clusterCentre = zeros(8, size(xTrainNorm,2));
        for angle = 1:8
            lowerIndex = 1 + numTrials*(angle-1);
            upperIndex = numTrials*angle;
            clusterCentre(angle,:) = mean(xTrainNorm(lowerIndex:upperIndex, :));
        end
    end
end
function [predAngle] = testAngleClassification(trial, tBeginTarget, tEndTarget, clusterCentre, meanXTrain)
    [metricPreMovementTest labels] = formatPreMovement(trial, tBeginTarget, tEndTarget);
    preCrtTest = metricPreMovementTest - meanXTrain;
    predAngle = nearestCentroid(clusterCentre, preCrtTest)';
    function [outputTrainMetric, outputLabel] = formatPreMovement(trials, tLow, tHigh)
        numTrials = size(trials,1);
        numAngle = size(trials,2);
        numNeuron = size(trials(1,1).spikes,1);
        numMetrics = 2;
        tDiff = tHigh - tLow; 
        N = numTrials*numAngle;
        D = tDiff*numNeuron;

        outputTrainMetric = zeros(numTrials, numAngle, numMetrics*numNeuron);
        outputLabel = zeros(numTrials, numAngle);

        for trial = 1:numTrials
            for angle = 1:numAngle
                metricStore = zeros(numNeuron, numMetrics);
                for neuron = 1:numNeuron
                    lenSpike = length(trials(trial,angle).spikes(neuron,:));
                    if lenSpike < tHigh-1
                        metricStore(neuron,:) = [mean(trials(trial,angle).spikes(neuron , tLow:lenSpike)), std(trials(trial,angle).spikes(neuron , tLow:lenSpike))];
                    else
                        metricStore(neuron,:) = [mean(trials(trial,angle).spikes(neuron , tLow:tHigh-1)), std(trials(trial,angle).spikes(neuron , tLow:tHigh-1))];
                    end                
                end
                outputTrainMetric(trial, angle, :) = metricStore(:);
                outputLabel(trial, angle) = angle;
            end
        end
        outputTrainMetric = reshape(outputTrainMetric, [N numMetrics*numNeuron]);
        outputLabel = reshape(outputLabel, [N 1]);
    end
    function [iProj] = nearestCentroid(centroids, xTest)
        [distProj iProj] = pdist2(centroids,xTest,'Euclidean' ,'Smallest', 1);
    end
end