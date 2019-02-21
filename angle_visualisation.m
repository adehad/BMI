clear all; close all; clc;
data = load('monkeydata_training.mat');
trials = data.trial;
[K, N] = size(trials);
I = 98; % number of neural units

colourGradChannel = [0 : 1/8 : 1-(1/8)]';
colours = [colourGradChannel, colourGradChannel, colourGradChannel];
colours(:,3) = linspace(0.5, 1, 8);

for n = 1:N
    for k = 1:K
        x = trials(k,n).handPos(1,:);
        y = trials(k,n).handPos(2,:);
        plot(x,y,'Color',colours(n,:)); hold on;
    end
end

grid on;
xlabel('x');
ylabel('y');
title('Hand position in the x-y plane')


%% Angle 
theta = zeros(K,N);
figure()

for n = 1:N
    for k = 1:K
        x = trials(k,n).handPos(1,:);
        y = trials(k,n).handPos(2,:);
        xvalid = x(301:end-100);
        yvalid = y(301:end-100);
        theta(k,n) = atan2(y(end), x(end)) + pi;
        plot(n,theta(k,n),'*','Color',colours(n,:)); hold on;
    end
end
