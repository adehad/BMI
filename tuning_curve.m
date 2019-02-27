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

% reaching angles: 30/180pi, 70/180pi, 110/180pi, 150/180pi, 190/180pi,
%                  230/180pi,
%                  310/180pi, 350/180pi
% NOTE: 270/180 is intentionally not here

% The actual duration of each angle event is not consistent

% The end goal is for a prediction of the next position, we will be given
% incrementally more data (in 20ms chunks) and be asked to predict the next
% position
%% Initialise some variables

% Reaching Angle set
angleInc = (40/180)*pi;
reachAngles = [(30/180)*pi:angleInc:(350/180)*pi];
reachAngles = reachAngles(reachAngles ~= (270/180)*pi );

% Time related
timeStep = 1e-3; % 1ms

%%
regionOfInterest = [300, 100]; % (1): starting sample, (2) samples to subtract from end

% pre-allocate for speeeeed
tuningCurve.meanMat = cell(size(trial)); % rows are trials, columns are angles
tuningCurve.stdMat = cell(size(trial));

% Go through region of interest and store mean and variance per trial
for incTrial=1:size(trial,1)
    for incAngle=1:size(trial,2)
        % Store Mean
        tuningCurve.meanMat{incTrial,incAngle} = ...
            mean( trial(incTrial,incAngle).spikes(:,regionOfInterest(1):end-regionOfInterest(2)) ,2);
        % Store Standard Deviation
        tuningCurve.stdMat{incTrial,incAngle} = ...
            std( trial(incTrial,incAngle).spikes(:,regionOfInterest(1):end-regionOfInterest(2))' ,1)'; % need to do some weird transpose to get it to work
    end
end

% pre-allocate for speeeeed
tuningCurve.mean = zeros(size(trial(1,1).spikes,1),size(trial,2)); % neurons are rows, columns are angles
tuningCurve.std  = zeros(size(trial(1,1).spikes,1),size(trial,2));

for incNeuron=1:size(trial(1,1).spikes,1)
    for incAngle=1:size(trial,2)
        
        tempMean = 0; tempStd = 0;
        for incTrial=1:size(trial,1)
            tempMean = tempMean + tuningCurve.meanMat{incTrial,incAngle}(incNeuron);
            tempStd = tempStd + tuningCurve.stdMat{incTrial,incAngle}(incNeuron);
        end
        
        % Store Mean
        tuningCurve.mean(incNeuron,incAngle) = tempMean/incTrial;
        % Store Standard Deviation
        tuningCurve.std(incNeuron,incAngle) = tempStd/incTrial;
        
    end
        tuningCurve.stdMax(incNeuron) = max(tuningCurve.std(incNeuron,:));
end

%% Plot 
fH = {}; % reset figure handle cell array

neuronSel = 1:10; 
stdMaxFactor = 0.9; % allow upto 90% of std max

% Tuning Curves
for jj=neuronSel
    fH{length(fH)+1} = figure;
        errorbar(rad2deg(reachAngles), tuningCurve.mean(jj,:),tuningCurve.std(jj,:))
        maxEl = find(tuningCurve.std(jj,:)>=stdMaxFactor*tuningCurve.stdMax(jj));
        hold on
        errorbar(rad2deg(reachAngles(maxEl)), tuningCurve.mean(jj,maxEl),tuningCurve.std(jj,maxEl),'o')

%     hold on;
%     yyaxis left
%     plot(tuningCurve.mean(jj,:))
%     yyaxis right
%     plot(tuningCurve.std(jj,:))
%     hold off
        xLims = xlim;
        xLims(2) = xLims(2) + rad2deg(reachAngles(1));
        xlim([xLims])
        xticks(rad2deg(reachAngles))
        grid on; grid minor
        title(sprintf('Neuron %d',jj))
        xlabel('angle (degrees)')
end
