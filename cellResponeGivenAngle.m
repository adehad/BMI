function sumCol = cellResponeGivenAngle(neuron, angle)

load('monkeydata_training.mat')

% assume recording starts at the same time. find maximum time duration (maxT)
maxT = 0;
len = zeros(1,100);

for i = 1:100  
    len(i) = length(trial(i, angle).spikes(neuron,:));
    if len(i) > maxT
        maxT = len(i);
    end   
end

% initialise zeros matrix of 100 x maxT
out = zeros(100, maxT);

% store one cell's spikes for each trial of given angle 
for i = 1:100
    out(i, 1:len(i)) = trial(i, angle).spikes(neuron,:);
end

% uncomment to plot rastor plot for all trials for given neuron
imagesc(out)

% sum down the columns to find total at each time for given cell and given
% direction
sumCol = sum(out);

end