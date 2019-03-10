% clear variables; close all;
rng(4);
% Loads data
load('monkeydata_training.mat');
trials = trial;
clear trial;
[train label N D aveTrain] = formatData(trials);
[xTrain lTrain xTest lTest] = splitTrain( train, label, N, 98 , 8);

meanXTrain = mean(xTrain,1);
A = xTrain - meanXTrain;
S = cov(A);

%image(reshape(meanXTrain, 98, 250)*100);
%% 
%[eigenvectors eigenvalues] = eig(S);
load('eigenneuron_24500.mat','eigenvectors','eigenvalues');
%%
% eigenvalues = sum(eigenvalues,2)';
[dSorted, iSorted] = sort(eigenvalues, 'descend');
vSorted = eigenvalues(:, iSorted);
plot(dSorted,'k','linewidth',1.5); grid on;
% Pick top eigenvalues
iSelected = iSorted(1:10);
eigSelected = eigenvectors(:, iSelected);
projTrain = A*eigSelected;

%% project
projTest = (xTest - meanXTrain)*eigSelected;

%% Nearest
[distProj iProj] = pdist2(projTrain,projTest,'Euclidean' ,'Smallest', 640);
lPred = label(iProj(1,:));

catAcc = sum(lPred == lTest)/ length(lTest);




function [xTrain lTrain xTest lTest] = splitTrain( data, label, N, D, class)
    randIdx = randperm(100, 20);
    %idx = (0:class-1)*100 + randIdx
    idx = [];
    for i = 1:class
        %Partition into sets
        idx = [idx (i-1)*100 + randIdx];
    end
    num = 1:N;
    num(idx) = 0;
    num = num(num ~= 0);
    xTrain = data(num, :);
    lTrain = label(num);
    xTest = data(idx, :);
    lTest = label(idx, :);
end
function [outputTrain outputLabel N D outputTrainAve] = formatData(trials)
    numTrials = size(trials,1);
    numAngle = size(trials,2);
    numNeuron = size(trials(1,1).spikes,1);
    tLow = 300;
    tHigh = 550;
    tDiff = tHigh - tLow; 
    N = numTrials*numAngle;
    D = tDiff*numNeuron;
    
    outputTrain = zeros(numTrials, numAngle, D); %N xD
    outputTrainAve = zeros(numTrials, numAngle, numNeuron); %N xD
    outputLabel = zeros(numTrials, numAngle);

    for trial = 1:numTrials
        for angle = 1:numAngle
            spikePattern = zeros(numNeuron, tDiff);
            for neuron = 1:numNeuron
                spikePattern(neuron ,:) = trials(trial,angle).spikes(neuron , tLow:tHigh-1);
                outputTrainAve(trial, angle, neuron) = sum(trials(trial,angle).spikes(neuron , tLow:tHigh-1));
            end
            outputTrain(trial, angle, :) = spikePattern(:);
            outputLabel(trial, angle) = angle;
        end
    end
    
    outputTrain = reshape(outputTrain, [N D]);
    outputLabel = reshape(outputLabel, [N 1]);
    outputTrainAve = reshape(outputTrainAve, [N numNeuron]);
end