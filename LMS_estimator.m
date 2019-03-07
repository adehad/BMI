%%
clear variables; clc;
% Load Data
% load monkeydata_training.mat
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

%% Load new feature space and training data
load('featureExtractedData.mat') 
    % outputTrain
    % 800 = 1 angle for 100 trials, 8 times/angles
    % 196 is mean then std for 1 neuron, 98 times/neurons
    % 325, time series
load('yExtractedData.mat')
    % yTtrain
    % 800 = 1 angle for 100 trials, 8 times
    % 2 = ground truth x and y positions of hand t
    % 325, time series
%% Population Coding

neuroInteg = 20;
% offset = 1+neuroInteg;
offset = 1;
mu = 0.25;

neuronSel = [55,34,96,96,96,96,92,55];
% neuronSel = neuronSel(neuronSel ~= 96);

% Get angle variation with time

% Each neuron has an associated weight
% weights = ones(length(neuronSel),1);
% W = zeros( size(trial,1), size(trial,2), length(weights) );

clear model

% iterate over time
for incTrial=1%:100
    for incAngle =1:8
        elemOf = incTrial +100*(incAngle-1);
        % Calculate the hand angle for each point in the trial
        ref.x = squeeze( yTrain(elemOf,1,:) );
        ref.y = squeeze( yTrain(elemOf,2,:) );  
        
        % initialise weights
        model(elemOf).W.x = zeros(2*length(neuronSel),length(ref.x)); % 2 because we have std and mean
        model(elemOf).W.y = zeros(2*length(neuronSel),length(ref.y));
        
        W.x = model(elemOf).W.x;
        W.y = model(elemOf).W.y;
        
        % initialise some vars - technically ref.x and y are the same length
        err.x = zeros(1,length(ref.x));
        err.y = zeros(1,length(ref.y));
        pred.x = zeros(1,length(ref.x));
        pred.y = zeros(1,length(ref.y));
        
        % LMS weight update
        for n=offset:length(ref.x)
            
            % initialise first prediction with bad weights
            for jj=1:length(neuronSel) % using selected neurons
                for kk=[-1,0] % for std (kk=-1) and mean(kk=0) in 196
                    pred.x(n) = pred.x(n) + W.x(2*jj+kk,n)*outputTrain(elemOf,2*neuronSel(jj)+kk,n);
                    pred.y(n) = pred.y(n) + W.y(2*jj+kk,n)*outputTrain(elemOf,2*neuronSel(jj)+kk,n);
                end
            end
            
            % prediction error n, e(n)
            err.x(n) = ref.x(n) - pred.x(n);
            err.y(n) = ref.y(n) - pred.y(n);
            
            % weights update rule
            for jj=1:length(neuronSel) % using selected neurons
                for kk=[-1,0] % for std (kk=-1) and mean(kk=0) in 196
                    W.x(2*jj+kk,n+1) = W.x(2*jj+kk,n) + mu*err.x(n)*...
                                        outputTrain(elemOf,2*neuronSel(jj)+kk,n);
                                  
                    W.y(2*jj+kk,n+1) = W.y(2*jj+kk,n) + mu*err.y(n)*...
                                        outputTrain(elemOf,2*neuronSel(jj)+kk,n);
                end
            end            
        end
        % Store model analysis for this trial
        model(elemOf).W = W;
        model(elemOf).err = err;
        model(elemOf).pred = pred;
        model(elemOf).ref = ref;
    end
end

%%
whatTrial = 1;
whatAngle = 1;
elemOf = whatTrial +100*(whatAngle-1);

N = length(model(elemOf).ref.x);
figure
 subplot(2,1,1)
     plot(1:N,model(elemOf).ref.x,'DisplayName','ref'); hold on;
     plot(1:N,model(elemOf).pred.x,'DisplayName','pred'); 
     title(sprintf('X Pos, Trial %d, Angle %d',whatTrial, whatAngle))
     ylabel('angle (degrees)')
     legend('show','Location','northwest')
 subplot(2,1,2)
     plot(1:N,model(elemOf).ref.y,'DisplayName','ref'); hold on;
     plot(1:N,model(elemOf).pred.y,'DisplayName','pred');
     title(sprintf('Y Pos Trial %d, Angle %d',whatTrial, whatAngle))
     ylabel('angle (degrees)')
     legend('show','Location','northwest')
     %%
 figure
 subplot(2,1,1)
     plot(1:N+1,model(elemOf).W.x);
     title(sprintf('X Pos Weights, Trial %d, Angle %d',whatTrial, whatAngle))
     ylabel('weight (mag)')
%      legend('show','Location','northwest')
 subplot(2,1,2)
     plot(1:N+1,model(elemOf).W.y);
     title(sprintf('Y Pos Weights, Trial %d, Angle %d',whatTrial, whatAngle))
     ylabel('weight (mag)')
%      legend('show','Location','northwest')

  