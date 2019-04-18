close all; clear all; clc;

% load pre-processed data
cumulativeSpikes = load('featureExtractedData.mat');
cumulativeSpikes = cumulativeSpikes.outputTrain;
handPosition = load('yExtractedData.mat');
handPosition = handPosition.yTrain;

% parameters
numNeuralUnits = 98;
numTrials = 100;
numAngles = 8;
numCoord = 2;
sequenceLen = size(cumulativeSpikes,3);
% global trainingLoss
trainingLoss = [];

% split into 80% training and 20% test sets
[xTrain, lTrain, xTest, lTest] = splitTrain(cumulativeSpikes,handPosition,numAngles);

% centre the datasets
ctrXTrain = xTrain - mean(xTrain, 1);
ctrLTrain = lTrain - mean(lTrain, 1);
ctrXTest = xTest - mean(xTrain, 1);
ctrLTest = lTest - mean(lTrain, 1);

% format for LSTM network
spikesTrain = cell(0.6*numTrials*numAngles,1);
posTrain = cell(0.6*numTrials*numAngles,1);
spikesValid = cell(0.2*numTrials*numAngles,1);
posValid = cell(0.2*numTrials*numAngles,1);
spikesTest = cell(0.2*numTrials*numAngles,1);
posTest = cell(0.2*numTrials*numAngles,1);

for i = 1 : 0.8*numTrials*numAngles
    spikesTrain{i,1} = squeeze(ctrXTrain(i,:,:));
    posTrain{i,1} = squeeze(ctrLTrain(i,:,:));
end

for j = 1 : 0.2*numTrials*numAngles
    spikesTest{j,1} = squeeze(ctrXTest(j,:,:));
    posTest{j,1} = squeeze(ctrLTest(j,:,:));
end

% LSTM architecture with 200 hidden units
numFeatures = 2*numNeuralUnits;
numHiddenUnits = 100;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numCoord)
    regressionLayer];

% Training options - Adam optimiser with 250 epochs, with a gradient
% threshold of 1 (prevent gradients from exploding), initial learning rate
% of 0.0025 and drop the learning rate after 75 epochs by multiplying by a
% factor of 0.5.
miniBatchSize = 20*numAngles;
options = trainingOptions('adam', ...
    'MaxEpochs',75, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'MiniBatchSize',miniBatchSize, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',37, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose',0, ...
    'ExecutionEnvironment','cpu', ...
    'OutputFcn',@(info)savetrainingplot(info), ...
    'Plots','training-progress');

% Train the LSTM network
net = trainNetwork(spikesTrain,posTrain,layers,options);

% %%
% net = load('net_1200.mat')
% net = net.net;
% 
% 
% %%
% miniBatchSize = 20*numAngles;
% LPred = predict(net,spikesTest,'MiniBatchSize',miniBatchSize);
% figure();
% time = 1 : sequenceLen;
% subplot(2,1,1);
% plot(LPred{4,1}(1,:), LPred{4,1}(2,:));
% hold on;
% plot(posTest{4,1}(1,:), posTest{4,1}(2,:));
% 
% 
% for i = 1 : 0.2*numTrials*numAngles
%     RMSE(i) = sqrt(mean(mean((LPred{i,1}-posTest{i,1}).^2)));
% end
% 

function info = savetrainingplot(info)
% trainingLoss = append(trainingLoss,
% trainingLoss =  [trainingLoss, info.TrainingRMSE];
TrainingRMSE = info.TrainingLoss;

if info.Epoch == 0
    save('TrainingRMSE.mat','TrainingRMSE');

else
    oldTrainingRMSE = load('TrainingRMSE.mat');
    oldTrainingRMSE = oldTrainingRMSE.TrainingRMSE;
    TrainingRMSE = [oldTrainingRMSE, TrainingRMSE];
    save('TrainingRMSE.mat','TrainingRMSE');
end

end

function [xTrain, lTrain, xTest, lTest] = splitTrain( data, label, class)
N = size(data,1);
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
xTrain = data(num, :, :, :);
lTrain = label(num, :, :);
xTest = data(idx, :, :, :);
lTest = label(idx, :, :);
end


