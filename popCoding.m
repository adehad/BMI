%%
clear variables; clc;
% Load Data
load monkeydata_training.mat
% 100 rows (trials), 8 columns (reaching angles)
% 98 neurons recorded from (98 channel), it reaches 182 times for each angle 
% 182*8 reach events
%  ~30% neurons are single unit, remaining multi-unit (MUA)
% 300ms before *movement* to 100ms after (400ms) of arm trajectory is in trial.handPos

% trial(n,k).spikes(i,:) == Trial n, k Reaching angle, neuron i
% .spikes() - each element represents a 1ms bin - if 1 a spike occured

% trial(n,k).handPos == Trial n, k Reaching angle
    % 3 rows: 1 = horizontal movement along screen plane  <->
    %         2 = vertical movement along screen plane  /\ and \/
    %         3 = perpendicular to screen - relatively small

% reaching angles: (30/180)pi, (70/180)pi, (110/180)pi, (150/180)pi, (190/180)pi,
%                  (230/180)pi,
%                  (310/180)pi, (350/180)pi
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
regionOfInterest = [300, 100]; % (1): movement onset sample, (2) samples to subtract from end

numNeurons = size(trial(1,1).spikes,1);
numAngles = size(trial,2);
numTrials = size(trial,1);

activitySeg = 20; % we use to split into batches of 20 elements

% pre-allocate for speeeeed
tuningCurve = cell(numNeurons,1); % an element for each neuron

lengthMatrix = zeros(size(trial)); % store the duration of everything

tempMean = 0; tempStd = 0;

% Go through region of interest and store mean and variance per trial
for incNeuron=1:numNeurons        % for each neuron
    for incAngle=1:numAngles      % for a given angle
        for incTrial=1:numTrials  % across all trials
            
            % Temporary Store spikes from region of interest
            tempSpikes = trial(incTrial,incAngle).spikes(incNeuron,regionOfInterest(1):end-regionOfInterest(2));
                        
            % Store duration of region of interest
            lengthMatrix(incTrial,incAngle) = length(tempSpikes);
            
            % Store Mean (i.e. average neural activity)
            tuningCurve{incNeuron}.meanMat(incTrial,incAngle) = ...
                mean( tempSpikes ,2);
            
            % Store Standard Deviation (i.e. changes in activity)
            tuningCurve{incNeuron}.stdMat(incTrial,incAngle) = ...
                std( tempSpikes, [] ,2); 
          
            % Temporary Store spikes from region of interest, add an extra
            % chunk of activitySeg of just before the region of interest
            tempSpikes = trial(incTrial,incAngle).spikes(incNeuron,regionOfInterest(1)-activitySeg-1:end-regionOfInterest(2));
            
            % nan-pad the tempSpikes variable (row vector)
            tempSpikes = [ tempSpikes, ...
                nan(1, activitySeg*ceil(length(tempSpikes)/activitySeg +1)-length(tempSpikes) ) ];
            
            % reshape the tempSpikes variable into batches of activitySeg
%             tempSpikes = reshape(tempSpikes, activitySeg, []); % OLD
                % now we take a moving window of size activitySeg
            tempSpikes2 = zeros(activitySeg, activitySeg*ceil(length(tempSpikes)/activitySeg));
            for ii=1:length(tempSpikes2)-activitySeg
                tempSpikes2(:,ii) = tempSpikes(ii:ii+activitySeg-1)';
            end
            
            % Store Mean in 20ms segments (i.e. current neural activity)            
            tuningCurve{incNeuron}.meanVec{incTrial,incAngle} = ...
                mean( tempSpikes2, 'omitnan' );
            
            % Store Standard Deviation in 20ms  (i.e. current variation in activity)
            tuningCurve{incNeuron}.stdVec{incTrial,incAngle} = ...
                std( tempSpikes2, 'omitnan' ); 
           
        end % end trial
    end % end angle
end % end neuron

% minT = shortest trial duration, 1=normal, 2=padded
minT(1) = min(lengthMatrix,[],'all');
minT(2) = activitySeg* ceil(minT(1)/activitySeg);

maxT(1) = max(lengthMatrix,[],'all');
maxT(2) = activitySeg* ceil(maxT(1)/activitySeg);


meanMatMean = zeros(numAngles, numNeurons);
stdMatMean  = zeros(numAngles, numNeurons);

% average across trials
for incNeuron=1:numNeurons        % for each neuron
    tuningCurve{incNeuron}.meanVecMean = zeros(numAngles, minT(1)+activitySeg);
    tuningCurve{incNeuron}.stdVecMean  = zeros(numAngles, minT(1)+activitySeg);
    for incAngle=1:numAngles      % for a given angle
        for incTrial=1:numTrials  % across all trials

            % Store Total Mean (i.e. average neural activity across trials)
            meanMatMean(incAngle,incNeuron) = ...
                 meanMatMean(incAngle,incNeuron) +  tuningCurve{incNeuron}.meanMat(incTrial,incAngle);
            
            % Store Total Standard Deviation (i.e. average changes in activity across trials)
            stdMatMean(incAngle,incNeuron) = ...
                 stdMatMean(incAngle,incNeuron) +  tuningCurve{incNeuron}.stdMat(incTrial,incAngle);
            
            % Store Total of the 20ms moving window (i.e. current neural activity) 
            % an extra +activitySeg is because we added the extra chunk earlier
            % which we use later to predict the 300ms point
            tuningCurve{incNeuron}.meanVecMean(incAngle,:) = ...
                 tuningCurve{incNeuron}.meanVecMean(incAngle,:) +  tuningCurve{incNeuron}.meanVec{incTrial,incAngle}(1:minT(1)+activitySeg);
             
            tuningCurve{incNeuron}.stdVecMean(incAngle,:) = ...
                 tuningCurve{incNeuron}.stdVecMean(incAngle,:) +  tuningCurve{incNeuron}.stdVec{incTrial,incAngle}(1:minT(1)+activitySeg);
             
            if incNeuron == 1 % only do this once
                % Store Total Tracjectory
                Traj.All(incTrial,incAngle,:,:) = trial(incTrial,incAngle).handPos(1:2,regionOfInterest(1):regionOfInterest(1)+minT(1)); %1=x,2=y
                Traj.All_padded(incTrial,incAngle,:,:) = [ trial(incTrial,incAngle).handPos(1:2,regionOfInterest(1):regionOfInterest(1)+lengthMatrix(incTrial,incAngle)), ...
                                                           nan(2,maxT(1)-lengthMatrix(incTrial,incAngle)) ]; %1=x,2=y
            elseif incNeuron == 2 % only do this once 
                Traj.mean = squeeze(mean(Traj.All,1));
                Traj.std  = squeeze(std(Traj.All,[],1));
                Traj.mean_padd = squeeze(mean(Traj.All_padded,1, 'omitnan'));
                Traj.std_padd  = squeeze(std(Traj.All_padded,[],1, 'omitnan'));
            end
        end % end trial
        
        % Convert Total to Averages
        meanMatMean(incAngle,incNeuron) = meanMatMean(incAngle,incNeuron) / incTrial;
        stdMatMean(incAngle,incNeuron) = stdMatMean(incAngle,incNeuron) / incTrial;
        tuningCurve{incNeuron}.meanVecMean(incAngle,:) = tuningCurve{incNeuron}.meanVecMean(incAngle,:) / incTrial;
        tuningCurve{incNeuron}.stdVecMean(incAngle,:) = tuningCurve{incNeuron}.stdVecMean(incAngle,:) / incTrial;
        
        
    end % end angle
end % end neuron


%% Neuron Selection
neuronList = 1:numNeurons;
stdMaxFactor = 0.85; % allow upto 85% of std max

% all neurons are good to start
goodNs = true(1,numNeurons);


% sort neurons by their standard deviation, highest first
[sorted.std , sorted.elem] = sort(stdMatMean,1,'descend');
    
% neuronSel = neuronList((tuningCurve.stdMax>1.5*mean(tuningCurve.stdMax)));
% determine selectivity
for ii=neuronList(goodNs) % for each neuron
    
    neuroTest = find( sorted.std(:,ii) >= stdMaxFactor*sorted.std(1,ii) );
    
    if length(neuroTest) < 3
        neuroTest = sorted.elem(neuroTest,ii);
        neuroTest = sort(neuroTest,1,'descend');
        if max(diff(neuroTest,1))>1 
            if min(diff(neuroTest,1))>1
                goodNs(ii) = false; % it a bad neuron
            end       
        end
    else
        goodNs(ii) = false; % it a bad neuron
    end
end

neuronSel = neuronList(goodNs);
neuronPref = []; % clear neuron direction preferences variable

for ii=1:length(neuronSel)
    
    neuroTest = find( sorted.std(:,neuronSel(ii)) >= stdMaxFactor*sorted.std(1,neuronSel(ii)) );
    prefAngle = sorted.elem(neuroTest(1),neuronSel(ii));
    neuronPref(ii) = sum( rad2deg(reachAngles(prefAngle)) .* ...
                          ( (eps+stdMatMean(prefAngle,neuronSel(ii))) / (eps+sum(stdMatMean(prefAngle,neuronSel(ii)))) )' );
                      % +eps prevents when std is zero
end


%% Plot 
fH = {}; % reset figure handle cell array

plotTuning = false;
plotTraj = false;
plotHist = false;

if plotTuning
    % Tuning Curves
    for jj=unique(neuronSel)
        fH{length(fH)+1} = figure;
            errorbar(rad2deg(reachAngles), meanMatMean(:,jj), stdMatMean(:,jj))
            maxEl = find(stdMatMean(:,jj)>=stdMaxFactor*sorted.std(1,jj) );
            hold on
            errorbar(rad2deg(reachAngles(maxEl)), meanMatMean(maxEl,jj),stdMatMean(maxEl,jj),'o')

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
end

if plotTraj
%     fH{length(fH)+1} = figure;
        for jj=1:numAngles
%             subplot(ceil(numAngles/2),2,jj)
            fH{length(fH)+1} = figure;
%             errorbar(squeeze(Traj.mean(jj,1,:)), squeeze(Traj.mean(jj,2,:)), squeeze(Traj.std(jj,2,:)),squeeze(Traj.std(jj,2,:)), squeeze(Traj.std(jj,1,:)),squeeze(Traj.std(jj,1,:)))
            hold on
            plotH(1) = errorbar(squeeze(Traj.mean_padd(jj,1,:)), squeeze(Traj.mean_padd(jj,2,:)), ... Mean
                                squeeze(Traj.std_padd(jj,2,:)),squeeze(Traj.std_padd(jj,2,:)), squeeze(Traj.std_padd(jj,1,:)),squeeze(Traj.std_padd(jj,1,:)), ... Standard Deviations
                                'o','LineWidth',0.25); % some parameters
            
            plotH(2) = plot(squeeze(Traj.mean(jj,1,1)), squeeze(Traj.mean(jj,2,1)),'m*','MarkerSize',25);
            plotH(3) = plot(squeeze(Traj.mean(jj,1,:)), squeeze(Traj.mean(jj,2,:)),'r--');
            plotH(4) = plot(squeeze(Traj.mean_padd(jj,1,:)), squeeze(Traj.mean_padd(jj,2,:)),'k:');
            xLims = xlim;
            
            legend(plotH,{'Full','Start','Min','Full'},'Location','best');
            
%             plot(xLims,[rad2deg(reachAngles(jj)), rad2deg(reachAngles(jj))])
            
%             xLims(2) = xLims(2) + rad2deg(reachAngles(1));
%             xlim([xLims])
%             xticks(rad2deg(reachAngles))
            grid on; grid minor
            title( sprintf('Reach Angle %02d',round(rad2deg(reachAngles(jj)))) )
            xlabel('x pos')
            ylabel('y pos')
        end
end

if plotHist
    histogram(neuronPref,rad2deg(reachAngles));
    title('Number of Neurons for each Reach Angle')
    xlabel('reach angle')
    ylabel('occurences')
end
%% Population Coding

% neuronSel = 1:numNeurons;
% neuronPref = rad2deg(reachAngles( sorted.elem(1,neuronSel) ));
mu = 0.9;

% Get angle variation with time


% reference signals 
ref.x = squeeze(Traj.mean(:,1,:));
ref.y = squeeze(Traj.mean(:,2,:));

% duplicate the restion position so the algorithm can converge to it first
ref.x = [ones(numAngles,activitySeg-1).*ref.x(:,1) ref.x];
ref.y = [ones(numAngles,activitySeg-1).*ref.y(:,1) ref.y];

clear pred err

gdNeur = length(neuronSel); % number of good neurons

% dataLen = minT(1)+activitySeg;
dataLen = length(ref.x);

% iterate over time
for incAngle = 1:numAngles  
%     for n=1:dataLen % discrete time increment
        
        % Each neuron has an associated weight
        W.x{incAngle} = zeros( length(neuronSel), dataLen );
        W.y{incAngle} = zeros( length(neuronSel), dataLen);
            % rows neurons, time delay 1 only
            % columns be time
        
        refX = ref.x(incAngle,:);
        refY = ref.y(incAngle,:);
        
        % initialise some vars
        predX = zeros(size(refX));
        predY = predX;
        errX   = predX; errY   = predX;
        
        % LMS weight update
        for n=1:dataLen
            
            % initialise first prediction with bad weights
            for jj=1:gdNeur % using selected neurons
                predX(n) = predX(n) + W.x{incAngle}(jj,n)*tuningCurve{neuronSel(jj)}.meanVecMean(incAngle,n) ...
                                                          *cosd(neuronPref(jj))*(activitySeg) ...
                                                          ;
                predY(n) = predY(n) + W.y{incAngle}(jj,n)*tuningCurve{neuronSel(jj)}.meanVecMean(incAngle,n) ...
                                                          *sind(neuronPref(jj))*(activitySeg) ...
                                                          ;
            end
            predX(n) = predX(n)/length(neuronSel);
            predY(n) = predY(n)/length(neuronSel);
            
            % prediction error n, e(n)
            errX(n) = refX(n) - predX(n);
            errY(n) = refY(n) - predY(n);
            
            % weights update rule
            for jj=1:gdNeur % using selected neurons
                W.x{incAngle}(jj,n+1) = W.x{incAngle}(jj,n) + mu * errX(n) * tuningCurve{neuronSel(jj)}.meanVecMean(incAngle,n) ...
                                                    *cosd(neuronPref(jj))*(activitySeg) ...
                                                    ;
                W.y{incAngle}(jj,n+1) = W.y{incAngle}(jj,n) + mu * errY(n) * tuningCurve{neuronSel(jj)}.meanVecMean(incAngle,n) ...
                                                    *sind(neuronPref(jj))*(activitySeg) ...
                                                    ;
            end
            
        end
        % Store weights for this trial
        % cols = angles. rows = time.
        pred.x(incAngle,:) = predX;
        pred.y(incAngle,:) = predY;
         err.x(incAngle,:) = errX;
         err.y(incAngle,:) = errY;
%     end
end

%%
plotLMS = true;

whatAngle = 1;  % max value = numAngles
whatNeuron = 1; % max value = length(neuronSel), can be an array, e.g. 1:5

if plotLMS
    figure; hold on
     plot(1:dataLen,W.x{whatAngle}(whatNeuron,2:end))
     plot(1:dataLen,W.y{whatAngle}(whatNeuron,2:end))
     title(sprintf('Neuron Weights against Time\n Neuron History Size = %d, Neuron = %s, Angle = %d',activitySeg, num2str(neuronSel(whatNeuron)), round(rad2deg(reachAngles(whatAngle))) ))
     ylabel('weight magnitude')
     xlabel('time index')
     legend({'x','y'},'Location','best')
    figure; hold on
      plot(1:dataLen,err.x(whatAngle,:))
      plot(1:dataLen,err.y(whatAngle,:))
      title(sprintf('Prediction Error against Time\n Neuron History Size = %d, Angle = %d',activitySeg, round(rad2deg(reachAngles(whatAngle))) ))
      ylabel('magnitude')
      xlabel('time index')
      legend({'x','y'},'Location','best')
    figure; hold on
      plot(ref.x(whatAngle,:),ref.y(whatAngle,:),'k:')
      plot(pred.x(whatAngle,:),pred.y(whatAngle,:))
      title(sprintf('Prediction vs Actual\n Neuron History Size = %d, Angle = %d',activitySeg, round(rad2deg(reachAngles(whatAngle))) ))
      xlabel('x pos')
      ylabel('y pos')
      legend({'real','pred'},'Location','best')
end