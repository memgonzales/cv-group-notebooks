import numpy as np
class KNearestNeighbor(object):
	def __init__(self):
		pass

	def train(self, X, y):
		"""
		For k-nearest neighbors training is just memorizing the training data.

		Inputs:
		- X: A numpy array of shape (num_train, D) containing the training data
		  consisting of num_train samples each of dimension D.
		- y: A numpy array of shape (N,) containing the training labels, where
			 y[i] is the label for X[i].
		"""
		self.X_train = X
		self.y_train = y
	
	def compute_distances_two_loops(self, X):
		"""
		Compute the distance between each test point in X and each training point
		in self.X_train using a nested loop over both the training data and the 
		test data.

		Inputs:
		- X: A numpy array of shape (num_test, D) containing test data.

		Returns:
		- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
		  is the Euclidean distance between the ith test point and the jth training
		  point.
		"""
		
		# X is test
		# self.X_train is train
		import numpy as np
		
		test_size = X.shape[0]
		train_size = self.X_train.shape[0]
		dists = np.empty((test_size, train_size))
		
		### START CODE HERE ###

		# This double-loop (naive) approach uses a literal implementation of the formula for Euclidean distance.
		for i in range(test_size):
			for j in range(train_size):
				dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
				
		### END CODE HERE ###
				
		return dists
	
	
	def predict_labels(self, dists, k=1):
		"""
		Given a matrix of distances between test points and training points,
		predict a label for each test point.

		Hint: Look up the function numpy.argsort.

		Inputs:
		- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
		  gives the distance betwen the ith test point and the jth training point.

		Returns:
		- y: A numpy array of shape (num_test,) containing predicted labels for the
		  test data, where y[i] is the predicted label for the test point X[i].  
		""" 
		
		test_size = dists.shape[0]
		y_pred = np.zeros(test_size)
		
		
		### START CODE HERE ###

		for i in range(test_size):
			# List storing the labels of the k-nearest neighbors
			knn = self.y_train[np.argsort(dists[i])[:k]]

			# Find the most recurring among the labels of the k-nearest neighbors
			y_pred[i] = np.argmax(np.bincount(knn))
		
		### END CODE HERE ###
		
		return y_pred
		
	def compute_distances_one_loop(self, X):
		# Same with compute_distances_two_loops
		test_size = X.shape[0]
		train_size = self.X_train.shape[0]
		dists = np.empty((test_size, train_size))
		
		### START CODE HERE ###

		# This single-loop approach takes advantage of NumPy's support for array slicing and element-wise operations.
		for i in range(test_size):
			dists[i] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis = 1))
		
		### END CODE HERE ###
				
		return dists

	def compute_distances_no_loops(self, X):
		test_size = X.shape[0]
		train_size = self.X_train.shape[0]
		dists = np.empty((test_size, train_size))
		
		### START CODE HERE ###

		# Recall that the Euclidean distance between A = [a_1, a_2, ..., a_n] and B = [b_1, b_2, ..., b_n]
		# is given by sqrt of summation of (a_i - b_i)^2 from i = 1 to n

		# Observe that (a_i - b_i)^2 = (a_i)^2 - 2(a_i)(b_i) + (b_i)^2
		#                            = (a_i)^2 + (b_i)^2 - 2(a_i)(b_i)

		# Assuming that the index of summation i goes from 1 to n,
		# summation of (a_i - b_i)^2 = summation of (a_i)^2 + summation of (b_i)^2 - 2 * summation of (a_i)(b_i)
		#                            = summation of (a_i)^2 + summation of (b_i)^2 - 2 * dot product of A and B

		# The sketch above shows the derivation of a mathematical formula to calculate the Euclidean distance
		# for one pair of samples A and B.

		# This no-loop approach extends this idea to matrices where the rows correspond to the samples.
		# Hence, it is necessary to perform some matrix transpositions, as seen in the implementation below:
		
		dists = np.sqrt(np.sum(X ** 2, axis = 1, keepdims = True) + np.sum(self.X_train ** 2, axis = 1) - 2 * (X @ np.transpose(self.X_train)))
				
		### END CODE HERE ###
		
		return dists
		
		#final should 500 x 5000
		
		#pass #INSERT CODE HERE
