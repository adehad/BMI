% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%               BE9-MBMI Brain-Machine Interfaces Spring 2019             %
% Team Name: Monkey Business                                              %
% Team Members: Adel Haddad, Aishwarya Pattar, Alex Dack, Shafa Balaram   %
% Implementation of a long short-term memory (LSTM) network in the        %
% prediction of hand position [x,y] using the mean and standard deviation %
% of the cumulative spikes across the time segment when there is movement.%
% Note: The Deep Learning Toolbox is required and the code was written    %
% using Matlab R2018b.                                                    %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

close all; clear variables; clc;

% start timer
tic

% load pre-processed data
cumulativeSpikes = load('featureExtractedData.mat');
cumulativeSpikes = cumulativeSpikes.outputTrain;
handPosition = load('yExtractedData.mat');
handPosition = handPosition.yTrain;

% parameters
selNeurons = [2 3 6 8 10 12 17 19 31 33 38 39 41 42 44 46 49 51 ...
    52 64 69 73 74 75 76 77 82 84 88 90 92 93 94 95 98];
index = [selNeurons*2-1; selNeurons*2];
index = reshape(index, size(index,1)*size(index,2), 1);
cumulativeSpikes = cumulativeSpikes(:, index, :);

numNeuralUnits = length(selNeurons);
numTrials = 100;
numAngles = 8;
numCoord = 2;
sequenceLen = size(cumulativeSpikes,3);

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

% LSTM architecture with 100 hidden units
numFeatures = 2*numNeuralUnits;
numHiddenUnits = 100;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numCoord)
    regressionLayer];

% Training options - Adam optimiser with 300 epochs, with a gradient
% threshold of 1 (prevent gradients from exploding), initial learning rate
% of 0.0025 and drop the learning rate after 75 epochs by multiplying by a
% factor of 0.5.
miniBatchSize = 20*numAngles;
options = trainingOptions('adam', ...
    'MaxEpochs',300, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'MiniBatchSize',miniBatchSize, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',50, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose',0, ...
    'ExecutionEnvironment','cpu', ...
    'OutputFcn',@(info)savetrainingplot(info), ...
    'Plots','training-progress');

% Train the LSTM network
net = trainNetwork(spikesTrain,posTrain,layers,options);

% stop timer
t = toc;
display(t)

% save network
save('net.mat', 'net')

% net = load('net.mat');
% net = net.net;
miniBatchSize = 20*numAngles;
LPred = predict(net,spikesTest,'MiniBatchSize',miniBatchSize);
fig1 = figure('Color', [1,1,1]);
set(gcf,'units','centimeters','Position',[1 1 20 15])
time = 1 : sequenceLen;
plot(LPred{1,1}(1,:), LPred{1,1}(2,:), 'linewidth', 1.5);
hold on;
plot(posTest{1,1}(1,:), posTest{1,1}(2,:), 'linewidth', 1.5);
grid on; grid minor
set(gca, 'FontName', 'Times', 'FontSize', 20)
l = legend('original, zero-mean signal', 'predicted signal');
set(l,'fontsize', 20, 'interpreter', 'latex', 'location', 'southeast');
xlabel('Horizontal Displacement, $x$','FontSize',22,'Interpreter','latex');
ylabel('Vertical Displacement, $y$','FontSize',22,'Interpreter','latex');
title(['\bf{Prediction of Hand Postion}', newline, ...
    '\bf{from Neural Spikes using an LSTM}'],'FontSize',22,'Interpreter','latex');
% save figure
set(fig1, 'PaperSize',[21 16], ...
    'DefaultFigurePaperUnits', 'centimeters', ...
    'DefaultFigureUnits', 'centimeters', ...
    'DefaultFigurePaperSize', [21, 16], ...
    'DefaultFigurePosition', [1, 1, 20, 15])
print(fig1, 'lstm_pred', '-dpdf', '-r400')

% Performance evaluation using the RMSE between the original centred and
% predicted signals
MSE = zeros(0.2*numTrials*numAngles,1);
for i = 1 : 0.2*numTrials*numAngles
    MSE(i) = mean(mean((LPred{i,1}-posTest{i,1}).^2));
end
RMSE = sqrt(sum(MSE)/(0.2*numTrials*numAngles));
display(RMSE)

TrainingRMSE = load('TrainingRMSE.mat');
TrainingRMSE = TrainingRMSE.TrainingRMSE;
iter = 1200;
fig2 = figure('Color', [1,1,1]);
set(gcf,'units','centimeters','Position',[1 1 20 15])
plot([1:iter+1], TrainingRMSE, 'linewidth', 1.5);
xlim([1, iter])
grid on; grid minor
set(gca, 'FontName', 'Times', 'FontSize', 20)
xlabel('Iteration','FontSize',22,'Interpreter','latex');
ylabel('RMSE','FontSize',22,'Interpreter','latex');
title(['\bf{Evolution of Training Loss (RMSE)}', newline, ...
    '\bf{with Number of Iterations}'],'FontSize',22,'Interpreter','latex');
% save figure
set(fig2, 'PaperSize',[21 16], ...
    'DefaultFigurePaperUnits', 'centimeters', ...
    'DefaultFigureUnits', 'centimeters', ...
    'DefaultFigurePaperSize', [21, 16], ...
    'DefaultFigurePosition', [1, 1, 20, 15])
print(fig2, 'lstm_loss', '-dpdf', '-r400')

% function to store training loss (RMSE)
function info = savetrainingplot(info)

TrainingRMSE = info.TrainingRMSE;

if info.Epoch == 0
    save('TrainingRMSE.mat','TrainingRMSE');
    
else
    oldTrainingRMSE = load('TrainingRMSE.mat');
    oldTrainingRMSE = oldTrainingRMSE.TrainingRMSE;
    TrainingRMSE = [oldTrainingRMSE, TrainingRMSE];
    save('TrainingRMSE.mat','TrainingRMSE');
end

end

% function to split dataset into 80% training and 20% testing
function [xTrain, lTrain, xTest, lTest] = splitTrain( data, label, class)
N = size(data,1);
randIdx = randperm(100, 20);
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


