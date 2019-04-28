%%% Team Members: Adel Haddad, Aishwarya Pattar, Alex Dack, Shafa Balaram
%%% BMI Spring 2019 (Update 19th March 2019)

function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)

% **********************************************************
%
% You can also use the following function header to keep your state
% from the last iteration
%
% function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
%                 ^^^^^^^^^^^^^^^^^^
% Please note that this is optional. You can still use the old function
% declaration without returning new model parameters.
%
% *********************************************************

% - test_data:
%     test_data(m).trialID
%         unique trial ID
%     test_data(m).startHandPos
%         2x1 vector giving the [x y] position of the hand at the start
%         of the trial
%     test_data(m).decodedHandPos
%         [2xN] vector giving the hand position estimated by your
%         algorithm during the previous iterations. In this case, N is
%         the number of times your function has been called previously on
%         the same data sequence.
%     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
%     in this case, t goes from 1 to the current time in steps of 20
%     Example:
%         Iteration 1 (t = 320):
%             test_data.trialID = 1;
%             test_data.startHandPos = [0; 0]
%             test_data.decodedHandPos = []
%             test_data.spikes = 98x320 matrix of spiking activity
%         Iteration 2 (t = 340):
%             test_data.trialID = 1;
%             test_data.startHandPos = [0; 0]
%             test_data.decodedHandPos = [2.3; 1.5]
%             test_data.spikes = 98x340 matrix of spiking activity



% ... compute position at the given timestep.

% Return Value:

% - [x, y]:
%     current position of the hand
%isempty(modelParameters.predAngle) 
dataLen = length(test_data.spikes);

if dataLen<=320
    tBeginTarget = 1; tEndTarget = 300;
    [predAngle] = testAngleClassification(test_data, tBeginTarget, tEndTarget, modelParameters.clusterCentre, modelParameters.trainCentre);
    modelParameters.predAngle = predAngle;
%     disp(['Pred Angle: ' num2str(predAngle)]);
end

[x,y] = popCoding_estimator(test_data, modelParameters);
% fprintf('\nPredicted X:%.4f \t Y:%.4f \n',x,y)

modelParameters.lastX = x;
modelParameters.lastY = y;

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

%%% ---- Least Mean Square (LMS) Filter for Trajectory Prediction ---- %%%
function [x,y] = popCoding_estimator(trial, modelParameters)
    %%  Initialise some variables

    % Reaching Angle set
    angleInc = (40/180)*pi;
    reachAngles = [(30/180)*pi:angleInc:(350/180)*pi];
    reachAngles = reachAngles(reachAngles ~= (270/180)*pi );
    regionOfInterest = [300, 0]; % (1): movement onset sample, (2) samples to subtract from end
    numNeurons = length(modelParameters.neuronSel);
    numAngles = size(trial,2);
    numTrials = size(trial,1);
    activitySeg = 20; % we use to split into batches of 20 elements

    % pre-allocate for speed
    tuningCurve = cell(numNeurons,1); % an element for each neuron
    lengthMatrix = zeros(size(trial)); % store the duration of everything
    
    % Neuron Selection
    neuronSel = modelParameters.neuronSel;
    neuronPref = modelParameters.neuronPref;
    
    % data length
    entireLen = length(trial(1,1).spikes);
    if entireLen > 320
        regionOfInterest(1) = entireLen -activitySeg+1; % -activitySeg, becuase training outputs an extra weight
    end
    
    minT(1) = length(trial(1,1).spikes(1,regionOfInterest(1):end-regionOfInterest(2))); % we are only given 1 trial, hence shortest = the length of the only one

    % Go through region of interest and store mean and variance per trial
    for incNeuron=1:numNeurons        % for each neuron
        for incAngle=1:numAngles      % for a given angle
            for incTrial=1:numTrials  % across all trials

                % Temporary Store spikes from region of interest
                tempSpikes = trial(incTrial,incAngle).spikes(modelParameters.neuronSel(incNeuron),regionOfInterest(1):end-regionOfInterest(2));

                % Store duration of region of interest
                lengthMatrix(incTrial,incAngle) = length(tempSpikes);

                % Store Mean (i.e. average neural activity)
%                 tuningCurve{incNeuron}.meanMat(incTrial,incAngle) = ...
%                     mean( tempSpikes ,2);

                % Store Standard Deviation (i.e. changes in activity)
%                 tuningCurve{incNeuron}.stdMat(incTrial,incAngle) = ...
%                     std( tempSpikes, [] ,2);

                % Temporary Store spikes from region of interest, add two extra
                % chunks of activitySeg of just before the region of interest.
                % We want the 'activity' from 20ms before the start, so we need
                % an extra 20ms before that point (40ms) to calculate the 'activity'.
                % 20ms = activitySeg
                tempSpikes = trial(incTrial,incAngle).spikes(modelParameters.neuronSel(incNeuron),regionOfInterest(1)-2*activitySeg-1:end-regionOfInterest(2));

                % nan-pad the tempSpikes variable (row vector)
                tempSpikes = [ tempSpikes, ...
                    nan(1, activitySeg*ceil(length(tempSpikes)/activitySeg)-length(tempSpikes) ) ];

                % reshape the tempSpikes variable into batches of activitySeg
                %             tempSpikes = reshape(tempSpikes, activitySeg, []); % OLD
                % now we take a moving window of size activitySeg
                tempSpikes2 = zeros(activitySeg, activitySeg*ceil(length(tempSpikes)/activitySeg));
                for ii=1:lengthMatrix(incTrial,incAngle)+activitySeg
                    tempSpikes2(:,ii) = tempSpikes(ii:ii+activitySeg-1)';
                end

                % Store Mean in 20ms segments (i.e. current neural activity)
                tuningCurve{incNeuron}.meanVec{incTrial,incAngle} = ...
                    mean( tempSpikes2, 'omitnan' );

                % Store Standard Deviation in 20ms  (i.e. current variation in activity)
%                 tuningCurve{incNeuron}.stdVec{incTrial,incAngle} = ...
%                     std( tempSpikes2, 'omitnan' );

            end % end trial
        end % end angle
    end % end neuron

    % minT = shortest trial duration, 1=normal, 2=padded
%     minT(1) = min(min(lengthMatrix));
%     minT(2) = activitySeg* ceil(minT(1)/activitySeg);

%     maxT(1) = max(max(lengthMatrix));
%     maxT(2) = activitySeg* ceil(maxT(1)/activitySeg);

%     meanMatMean = zeros(numAngles, numNeurons);
%     stdMatMean  = zeros(numAngles, numNeurons);

    % average across trials
    for incNeuron=1:numNeurons        % for each neuron
        tuningCurve{incNeuron}.meanVecMean = zeros(numAngles, minT(1)+activitySeg);
%         tuningCurve{incNeuron}.stdVecMean  = zeros(numAngles, minT(1)+activitySeg);
        for incAngle=1:numAngles      % for a given angle
            for incTrial=1:numTrials  % across all trials

                % Store Total Mean (i.e. average neural activity across trials)
%                 meanMatMean(incAngle,incNeuron) = ...
%                     meanMatMean(incAngle,incNeuron) +  tuningCurve{incNeuron}.meanMat(incTrial,incAngle);

                % Store Total Standard Deviation (i.e. average changes in activity across trials)
%                 stdMatMean(incAngle,incNeuron) = ...
%                     stdMatMean(incAngle,incNeuron) +  tuningCurve{incNeuron}.stdMat(incTrial,incAngle);

                % Store Total of the 20ms moving window (i.e. current neural activity)
                % an extra +activitySeg is because we added the extra chunk earlier
                % which we use later to predict the 300ms point
                tuningCurve{incNeuron}.meanVecMean(incAngle,:) = ...
                    tuningCurve{incNeuron}.meanVecMean(incAngle,:) +  tuningCurve{incNeuron}.meanVec{incTrial,incAngle}(1:minT(1)+activitySeg);

%                 tuningCurve{incNeuron}.stdVecMean(incAngle,:) = ...
%                     tuningCurve{incNeuron}.stdVecMean(incAngle,:) +  tuningCurve{incNeuron}.stdVec{incTrial,incAngle}(1:minT(1)+activitySeg);


            end % end trial

            % Convert Total to Averages
%             meanMatMean(incAngle,incNeuron) = meanMatMean(incAngle,incNeuron) / incTrial;
%             stdMatMean(incAngle,incNeuron) = stdMatMean(incAngle,incNeuron) / incTrial;
            tuningCurve{incNeuron}.meanVecMean(incAngle,:) = tuningCurve{incNeuron}.meanVecMean(incAngle,:) / incTrial;
%             tuningCurve{incNeuron}.stdVecMean(incAngle,:) = tuningCurve{incNeuron}.stdVecMean(incAngle,:) / incTrial;

        end % end angle
    end % end neuron



    %% Population Coding
    
    clear pred

    gdNeur = length(neuronSel); % number of good neurons
    
    % Each neuron has an associated weight
    W = modelParameters.W;
    expFilterPrevWeight = 0.6;
   
    if entireLen > 320
        dataOffset = entireLen - 320;
        dataLen = minT(1);
    else
        dataOffset = entireLen - 320;
        dataLen = minT(1);
    end
    
    
    % iterate over time
    for incAngle = modelParameters.predAngle  
        physicalLimit = struct;
        physicalLimit.x = modelParameters.ref.x(incAngle, end);
        physicalLimit.y = modelParameters.ref.y(incAngle, end);
        physicalLimit.xStep = modelParameters.ref.xStep(incAngle);
        physicalLimit.yStep = modelParameters.ref.yStep(incAngle);
        limitOffset.x = 1.1;
        limitOffset.y = 1.1;

        % initialise some vars
        predX = zeros(1,dataLen);
        predY = predX;

        for n=1:dataLen
            if n < size(W.x{incAngle},2)-dataOffset
                for jj=1:gdNeur % using selected neurons
                    predX(n) = predX(n) + W.x{incAngle}(jj,dataOffset+n)*tuningCurve{jj}.meanVecMean(1,n) ...
                        *cosd(neuronPref(jj))*(activitySeg) ...
                        ;
                    predY(n) = predY(n) + W.y{incAngle}(jj,dataOffset+n)*tuningCurve{jj}.meanVecMean(1,n) ...
                        *sind(neuronPref(jj))*(activitySeg) ...
                        ;
%                     if abs(predY(n)) > 1e4
%                         warning('oops')
%                     end
                end
                predX(n) = predX(n)/length(neuronSel);
                predY(n) = predY(n)/length(neuronSel);

                if n>1
                    % Enforce typical physical step size
                    if abs(predX(n)-predX(n-1)) > limitOffset.x*abs(physicalLimit.xStep)*activitySeg
                        predX(n) =  predX(n-1) + limitOffset.x*physicalLimit.xStep*sign(predX(n)-predX(n-1))*activitySeg;
                    end

                    if abs(predY(n)-predY(n-1)) > limitOffset.y*abs(physicalLimit.yStep)*activitySeg
                        predY(n) =  predY(n-1) + limitOffset.y*physicalLimit.yStep*sign(predY(n)-predY(n-1))*activitySeg;
                    end

%                     Exponential Filtering
                    predX(n) = expFilterPrevWeight*predX(n-1) + (1-expFilterPrevWeight)*predX(n);
                    predY(n) = expFilterPrevWeight*predY(n-1) + (1-expFilterPrevWeight)*predY(n);
                end

                % Enforce physical limits
                if abs(predX(n)) >= limitOffset.x*abs(physicalLimit.x)
                    predX(n) =  limitOffset.x*physicalLimit.x;
                end

                if abs(predY(n)) >= limitOffset.y*abs(physicalLimit.y)
                    predY(n) =  limitOffset.y*physicalLimit.y;
                end
%             
            else
                % Stops predicting once we run out of weights
                % simply give the physical limit
                predX(n) =  limitOffset.x*physicalLimit.x;
                predY(n) =  limitOffset.y*physicalLimit.y;
                
            end
        end

        % Store weights for this trial
        % cols = time. rows = angles.
        pred.x(incAngle,:) = predX;
        pred.y(incAngle,:) = predY;

    end

    x = pred.x(incAngle,:);
    y = pred.y(incAngle,:);
    
    x = x(x ~=0); 
    y = y(y ~=0);
    
    if entireLen == 320
        x = trial.startHandPos(1);
        y = trial.startHandPos(2);
    else
        try
            x = x(end);
        catch
            x = modelParameters.lastX;
        end
        try 
            y = y(end);
        catch
            y = modelParameters.lastY;
        end
    end

    
end
