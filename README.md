# BMI

# Report 
" Your error will be the root mean squared (rms) error of the trajectories you compute. The mean will be taken across both dimensions, over all trajectories. The units of this error is in cm. "



Abstract
	- Have one sentence that describes each section of the report. <250 words


Introduction
	- Description of the monkey data 
	- Aim of what we are trying to achieve 


Approach to solving the problem/Methods
Combine these two categories??
	- How classification is used to identify the angle of movement
	- How LMS determines the weights given to neurones - I don't understand whats happening in this code
	- nLMS
	- Include the maths behind these to justify

*If we attempt other methods I don't know whether they should go in this section and result, or just be mentioned in discussion

Training used:
	Classification
	- Identified cluster centres 196dim feature space using feature extraction of cumulative mean and cumulative standard dev over premovement area for each angle
	- 98% accuracy - include graph for centroid 
	- Each cluster centre = one reaching angle
	- Compare with Euclidian distance
	
	One step ahead prediction using NLMS
	- In post movement area, look at prev 20ms of activity and take mean for each neuron per angle
	- Originally standard dev was used to select neurones rather than use all neurones
		○ Standard dev and threshold to select neurones (85%)
		○ Discarded any neurones which exceeded the threshold for multiple angles
		○ Georgopoulos - Primate Motor cortex and free arm movement … (1988) : population vector eq
	- Optimised code and then used all the neurones
	- For a given neuron have 1x8 (average of the trails per angle. Each element is struct containing all neuron activities) Calculating the standard deviation for a given neuron across all angles.for each neuron, there are 8 sd values (one for each angle) 
		○ Sort in descending order of SD
		○ Pick the max sd to identify the preferred angle
	- Calculate change in x and change in y of the average trajectory per angle and identify the max. also find endpoint of average trajectory per angle
	- NLMS with mean calculated in prev20ms. NLMS train weights to multiply by the mean. Weights for each angle. Output of the filter is the next position


Testing
	Classification
	- Classification (k-nearest centroid with k=1) - identifies which reaching angle

	- Using classified angle, select the weights to use from training model 
	- Given the last time value (last element) - calculate the mean of the last 20 elements for the selected neurones (we have selected all the neurons from training)
	- Multiply weights by the mean activity and divide by number of the selected neurones 
	- check if the change in x-y > max step size from training. If it is greater, cap at the max. *continuity argument - can't move faster than average movement*. Cap the max displacement to the man end point of trajectory from the training 
	- Stop predicting if we run out of weights - stay in the location 

	Validation
	- Other method: K-fold cross validation

Optimisation 
	- All neurones instead of selecting
	- Put caps in place 
	- Removed repeating classification
	- Removed repeating prediction - only analysing the latest 20ms sample with the 20ms from the end of the previous sample 


Results
	- RMSE in classification method
	- Table of RMS error of trajectories
	- Mean across both trajectories over all trajectories (in cm)
	- Plots of prediction vs real
	- Put confusion matrix here


Discussion
	- Performance of algorithm
		○ Talk about Confusion Matrix 
	- How it could be improved
		○ Inclusion of Deep Learning MATLAB toolbox - LSTM
			§ Blackbox, ages to tune parameters
		○ Assumption of premovement, rather than detecting??
			§ Didn’t use Fano factor to detect this 
	- Compare against other approaches
		○ Compare accuracy and time? This will give more references
		○ LSTM
		○ LMS vs nLMS
		○ Using averaging method
		
Might be good to rank each method in a table in terms of speed and RMSE and then explain any limitations 

References
	- Fano paper 
Do we need to reference the algorithms? (classification/nLMS)




-----------------------------------------------------------------------------------------------------------------------------




# TODOs
Dack: Confusion Matrix, Error Given the Mean Trajectory (RMSE), Make function selecting angle

Shafa: NLMS

Ash: Report (Overleaf), investigate code efficiency 

Adel: Make a function that takes an average trajectory and an angle - use LMS for some weights

# Further Discussions
- Bayes
- K-nearest centroid - uses the Fano Factor - to make an assumption of the movement decision. 
  - Captures population vector by using Fano factor for each neuron - indicates the neuron preference which is reflected in the accuracy of the k-nearest centroid
  - LMS captures finer tuning when comapred to the mean trajectory

# Approaches:
1. Eigen-neuron:
	1. Create a `98x1000` (Ash found max value less than 1e3) zero padded thing for eigenneurons
	2. Cut all 'images' from `300:550` (250 elements)
		1. Get's the steady state response - in a way - feat. Shafa. When the monkey does something
	3. Create an image for each trial and angle
	4. `D=(250x98)`, `N=800` `N=(numAngle*numTrials)`
	5. `X=DxN`,	`L = 1xN` - `L`=Labels
	6. `A = X - mean(X,2)` - recenter the image with the mean image
	7. `S = cov(A) = (A^T A)/(N)`
		1. `NxN` matrix
	8. `eig(S)`
	9. Sorting eigenvalues in descending order
	10. Pick the `m` strongest 
	11. Split into Train (80), Testing (20)



