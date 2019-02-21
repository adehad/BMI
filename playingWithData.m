%%

% Load Data
load monkeydata_training.mat
% 100 rows (trials), 8 columns (reaching angles)
% 98 neurons recorded from (98 channel), it reaches 182 times for each angle 
% 182*8 reach events
%  ~30% neurons are single unit, remaining multi-unit (MUA)
% 300ms before to 100ms after (400ms) of arm trajectory is in trial.handPos

% trial(n,k).spikes(i,:) == Trial n, k Reaching angle, neuron i
% .spikes() - each element represents a 1ms bin - if 1 a spike occured

% trial(n,k).handPos == Trial n, k Reaching angle
    % 3 rows: 1 = horizontal movement along screen plane  <->
    %         2 = vertical movement along screen plane  /\ and \/
    %         3 = perpendicular to screen - relatively small

% reaching angles: 30/180?, 70/180?, 110/180?, 150/180?, 190/180?,
%                  230/180?,
%                  310/180?, 350/180?
% NOTE: 270/180 is intentionally not here

% The actual duration of each angle event is not consistent
%% Initialise some variables

% Reaching Angle set
angleInc = (40/180)*pi;
reachAngles = [(30/180)*pi:angleInc:(350/180)*pi];
reachAngles = reachAngles(reachAngles ~= (270/180)*pi );

% Time related
timeStep = 1e-3; % 1ms

%%

clear all; close all;
 load('monkeydata_training.mat');
trials = trial;
%% trial: 100 trials by 8 reaching angles
% trial(1,1) - 
x = trials(1,1);

%% Raster plot
neural = x.spikes;
colormap('gray')
imagesc(~neural);
xlabel('Time (ms)');
ylabel('Neural Index');
title('Raster plot');

%% All trials
K = 98; timeWindowLen = 5; timeWindow = 1*10^(-3)*timeWindowLen; t = 1;
tLow = 300; tUpper = 100;
for angle = 1:8
DtIndex = timeWindowLen - 1;
%for t = 1:size(trial,1)
    trial = trials(t,angle);
    spikes = trial.spikes(:, :);
    lengthSpike = size(spikes, 2);
    spikes = spikes(:, tLow:lengthSpike - tUpper);
    timeUpperIndex = size(spikes,2)- DtIndex;
    spikeDensity = zeros(timeUpperIndex, 1);
    for time = 1:timeUpperIndex
        numSpikesWindow = spikes(:, time:DtIndex + time);
        numOccur = sum(numSpikesWindow,2);
        spikeDensity(time) = sum(numOccur,1)/(K*timeWindow);
    end
%end

%% Peri stimulus time histogram
figure;
subplot(2,1,1)
tAxis = [1:timeUpperIndex].*1*10^(-3);
plot(tAxis, spikeDensity); hold on; xlabel('Time (s)'); ylabel('Spike Density');
subplot(2,1,2)
plot(tAxis, trial.handPos(1:2,tLow:lengthSpike - tUpper - DtIndex));
end
