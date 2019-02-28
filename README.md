# BMI

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



