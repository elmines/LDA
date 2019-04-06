import numpy as np

num_predictors = 20
num_classes = 5
classes = [i for i in range(num_classes)]

## Generating Synetic Data
def random_covar(N):
	vals = np.random.randint(-5, 5, size=(N, N))
	return vals @ np.transpose(vals)
samples_per_class = [10, 5, 20, 8, 13]
true_covar = random_covar(num_predictors)
samples = []
for i in range(num_classes):
	mean = [np.random.randint(-50, 50) for _ in range(num_predictors)]
	samples.append( np.random.multivariate_normal(mean, true_covar, size=samples_per_class[i]) )
totalSamples = sum(len(pop) for pop in samples) 

## Pool covarianc matrices
def pooled_cov(pops):
	num_samples = np.array([ len(pop) - 1 for pop in pops])
	covs = [np.cov(pop, rowvar=False) for pop in pops]
	denom = sum(num_samples)
	num = sum( map(lambda pair: pair[0]*pair[1], zip(num_samples, covs)) )
	return num / denom

## Calculate model paremeters (estimated inverse covariance and class probabilities)
classMeans = np.zeros( (num_classes, num_predictors) )
for (i, pop) in enumerate(samples):
	classMeans[i, :] = np.mean(pop, axis=0)
classProbs = np.array([len(pop) for pop in samples]) / totalSamples
inverse_covar = np.linalg.inv( pooled_cov(samples) )

## Create predictor function
bias = np.log(classProbs)
for (i, mean) in enumerate(classMeans):
	bias[i] -= (1/2) * np.transpose(mean) @ inverse_covar @ mean
weight = classMeans @ inverse_covar
def scores(x):
	return weight @ x + bias
def predict(x):
	return np.argmax(scores(x))

correct = 0
for (i, pop) in enumerate(samples):
	for sample in pop:
		correct += predict(sample) == i
accuracy = correct/totalSamples
print("accuracy =", accuracy)
