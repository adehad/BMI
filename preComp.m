% key facts:
% trial is a STRUCT has dimensions 100 (trials) x 8 (reaching angles)
% each element of trial has trialID, spikes(98x628), and handPos(3x628)
% spike trains from 98 neural units, from single unit or multi-unit
%   neurones. 628 is 628ms recording
% monkey reached 182 times along each of the 8 different reaching angles
% shows monkey's arm trajectory on each trial between 300ms before onset of
%   movement to 100ms after the end of movement.
% handPos represents in x,y,z axes. hand position given in mm at each 1ms
% interval


load('monkeydata_training.mat')

%% part 1 - population raster plot for single trial
% https://www.mathworks.com/matlabcentral/fileexchange/45671-flexible-and-fast-spike-raster-plotting

% rastor plot for neural activity for trial(1,1)
figure()
imagesc(trial(1,1).spikes)
title('Population Rastor Plot for Single Trial');
xlabel('time (ms)');
ylabel('cells');

%% part2 - raster plot for one cell over many trials

neuron = 1; % row 1 of spikes for each trial.

% find max time duration for neuron in trials
max_duration = 0;
for i = 1:100
   for j = 1:8
       len = length(trial(i, j).spikes(neuron,:));
       
       if len > max_duration
           max_duration = len;
       end          
   end
end

% initialise matrix containing sum of neuron activitiy for each angle
neuron_activity = zeros(8,max_duration);

for angle = 1 : 8
   
    sumCol = cellResponeGivenAngle(neuron, angle);
    neuron_activity(angle, 1:length(sumCol)) = sumCol;
    
end

imagesc(imgaussfilt(neuron_activity,1));
