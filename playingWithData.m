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
