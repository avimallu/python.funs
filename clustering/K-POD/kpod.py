import sys
import pandas as pd
import numpy  as np

class k_pod:
    
    def __init__(self, max_iter=300, tol=0, seed=None):
        self.max_iter = max_iter
        self.tol      = tol
        self.seed     = seed
    
    def __euclidean_distance(self, point_1, point_2):
        return np.sum((point_1 - point_2)**2)

    def __initialize(self, data, n_clusters):
        """ Initialize cluster centers using k-means++ algorithm.
        Parameters
        ----------
        data: {array-like, sparse matrix} of shape (N, P)
            Data to predict clusters for.
        n_clusters: int
            The number of cluster centers to initialize.
        Returns 
        -------
        centroids: array of shape (n_clusters,)
            Coordinates for each cluster center.
        """
        # initialize 
        data = np.array(data)
        N = data.shape[0]

        # initialize the centroids list and add a randomly selected data point to the list
        centroids = []
        if self.seed is not None:
            np.random.seed(self.seed)
        centroids.append(data[np.random.randint(N), :])

        # compute remaining k - 1 centroids
        for cluster in range(n_clusters - 1):

            # initialize a list to store distances of data points from nearest centroid
            distances = []

            for data_idx in range(N):

                # save current data point's coordinates
                point = data[data_idx, :]
                dist = sys.maxsize

                # loop through each centroid to find the minimum distances 
                for centroid_idx in range(len(centroids)):

                    # compute distance of 'point' from each of the previously selected centroid and store the minimum distance
                    curr_distance = self.__euclidean_distance(point, centroids[centroid_idx])
                    dist = min(dist, curr_distance)

                # add distance to array
                distances.append(dist)

            # data point with max distance
            distances = np.array(distances)

            # add centroid to array and reset distances
            center = data[np.argmax(distances), :]
            centroids.append(center)
            distances = []

        # return array of centroids
        return centroids

    def __cluster_assignment(self, data, cluster_centers):
        """ Assign each point in the dataset to a cluster based on its distance from cluster centers

        This is a helper method for the main kPOD functionality. It 
        executes the cluster assignment part of the algorithm.
        Parameters
        ----------
        data: {array-like, sparse matrix} of shape (N, P)
            Data to predict clusters for.
        cluster_centers: {array-like, sparse matrix} of shape (K,)
            Central point of each of the K clusters.
        N: int
            The number of observations in the data.

        k: int
            The number of clusters to assign centers for.
        Returns 
        -------
        cluster_assignment: ndarray of shape (N,)
            The cluster index that each data point was assigned to.
        """

        # set empty distance array with length of num clusters
        cluster_assignment = np.zeros(self.N)
        dist = np.zeros(self.k)

        # iterate through observations
        for num in range(0, self.N):

            # iterate through each cluster
            for cluster in range(self.k):

                # assign distance between point and cluster center
                dist[cluster] = self.__euclidean_distance(data[num], cluster_centers[cluster])

            # assign point to cluster center with lowest distance
            cluster_assignment[num] = np.argmin(dist)

        # return the cluster assignments for this iteration
        return cluster_assignment

    def __move_centroids(self, data, cluster_centers, cluster_assignment):
        """ Move each cluster centroid to the mean location of the points that are assigned to it.

        This is a helper method for the main kPOD functionality. It 
        executes the move cluster centroids part of the algorithm.
        Parameters
        ----------
        data: {array-like, sparse matrix} of shape (N, P)
            Data to predict clusters for.
        cluster_centers: {array-like, sparse matrix} of shape (K,)
            Central point of each of the K clusters.
        cluster_assignment: {array-like, sparse matrix} of shape (N,)
            Array containing the cluster index that each data point was assigned to.
        Returns 
        -------
        cluster_assignment: ndarray of shape (N,)
            The cluster index that each data point was assigned to.
        """

        # iterate through each cluster 
        for num in range(1, self.k+1):

            # make empty array cluster points
            cluster_points = list()

            # iterate through each data point
            for i in range(0, self.N):

                # if the cluster is assigned to this centroid, add it to the list of cluster points
                if int(cluster_assignment[i]) == (num-1):

                    #  add data point to list of cluster points
                    cluster_points.append(data[i])

            # convert the cluster points to an ndarray
            cluster_points = np.array(cluster_points)

            # set the new cluster centroid location to the main of the points it is assigned to
            if cluster_points.shape[0] == 0: # if there are no points in the cluster
                cluster_centers[num-1] == np.nan # there is no need for that cluster centre
            else:
                cluster_centers[num-1] = cluster_points.mean(axis=0)

        # return moved cluster centers
        return cluster_centers

    def __check_convergence(self, cluster_centers, past_centroids):
        """ Ensure that each cluster center is within the tolerance level of the last centroid.

        This is a helper method for the main kPOD functionality. It 
        executes the check convergence part of the algorithm.
        Parameters
        ----------
        cluster_centers: {array-like, sparse matrix} of shape (K,)
            Central point of each of the K clusters.
        past_centroids: {array-like, sparse matrix} of shape (K,)
            Array containing central points from the last kPOD iteration.
        tol: float
            The tolerance for each cluster center and its past centroid.

        num_iters: int
            Number of iterations of the algorithm.
        Returns 
        -------
        centroids_complete: boolean
            True if the cluster centers have converged, False otherwise.
        """

        # if it is the first iteration, algorithm has not converged
        if self.num_iters == 0:
            return False

        # set initial complete to 0
        centroids_complete = 0

        # check if k-means is complete
        for i in range(len(cluster_centers)):

            # if the distance between this centroid and the past centroid is less than tolerance
            if (self.__euclidean_distance(cluster_centers[i], past_centroids[i]) <= self.tol):

                # add centroid to the list of complete centroids
                centroids_complete += 1

        # return list of centroids that have converged
        return centroids_complete

    def __fill_data(self, MISSING_DATA, cluster_centers, cluster_assignment):
        """ Fill missing data with the average values for each data point's cluster.

        This is a helper method for the main kPOD functionality. It 
        executes the fill data part of the algorithm.
        Parameters
        ----------
        MISSING_DATA: {array-like, sparse matrix} of shape (N,P)
            Data with missing values.
        cluster_centers: {array-like, sparse matrix} of shape (K,)
            Central point of each of the K clusters.
        cluster_assignment: {array-like, sparse matrix} of shape (N,)
            Array containing the cluster index that each data point was assigned to.
        Returns 
        -------
        filled_data: {array-like, sparse matrix} of shape (N,P)
            Data with all nan values filled.
        """

        # save filled data as copy of missing data
        filled_data = np.array(MISSING_DATA.copy())

        # iterate through missing data
        for i in range(len(filled_data)):

            # set current cluster as cluster assignment of this data point
            obs_cluster = int(cluster_assignment[i])

            # reset counter to 0
            j = 0

            # iterate through each value
            for val in filled_data[i]:

                # if value is empty, replace it with cluster center value
                if (np.isnan(val)):

                    # replace value with cluster center value (mean of its dimension)
                    filled_data[i][j] = cluster_centers[obs_cluster][j]

                # increment counter
                j+=1

        # return data with all nan values filled 
        return filled_data

    def k_pod(self, data, k):
        """ Compute cluster centers and predict cluster index for sample containing missing data.
        Parameters
        ----------
        data: {array-like, sparse matrix} of shape (N, P)
            Data to predict clusters for.
        Returns 
        -------
        labels: ndarray of shape (N,)
            Index of the cluster each sample belongs to.
        """
        # convert data to numpy array
        data   = np.array(data)
        self.k = k

        # assign initial variables
        self.N = data.shape[0]
        P = data.shape[1]
        self.num_iters = 0   

        # collect missing indiices
        MISSING_DATA = data.copy()

        # initialize past centroids
        past_centroids = []
        cluster_centers = []
        cluster_assignment = []

        # loop through max iterations of kPOD
        while self.num_iters < self.max_iter:

            """
            STEP 1: Imputation of missing values
            fill with the mean of the cluster (centroid)
            """

            # if it has been multiple iterations, fill with algorithm
            if self.num_iters > 0:

                # fill data after first iteration
                filled_data = self.__fill_data(MISSING_DATA, cluster_centers, cluster_assignment)

                # save data as np array
                filled_data = np.array(filled_data)

            # fill with initial imputation if first iteration 
            else:

                # initial imputation
                data_frame = pd.DataFrame(data)
                filled_data = np.array(data_frame.fillna(np.nanmean(data)))

                # initialize cluster centers so other methods work properly
                cluster_centers = self.__initialize(filled_data, self.k)

            """
            STEP 2: K-Means Iteration
            """

            # Cluster Assignment
            cluster_assignment = self.__cluster_assignment(filled_data, cluster_centers)

            # Move centroids
            cluster_centers = self.__move_centroids(filled_data, cluster_centers, cluster_assignment)

            """
            STEP 3: Check for convergence
            """

            # check for convergence of algorithm
            centroids_complete = self.__check_convergence(cluster_centers, past_centroids)  

            # set past centroids to current centroids  
            past_centroids = cluster_centers

            # increase counter of iterations
            self.num_iters += 1

            # if k means is complete, end algo
            if (centroids_complete):
                break

        # return assignments and centroids
        cluster_ret = {"ClusterAssignment" : cluster_assignment, "ClusterCenters" : cluster_centers}

        cluster_return  = (cluster_assignment, cluster_centers, filled_data)
        return cluster_return
    
    def log(self, x, base):
        return np.log(x) / np.log(base)
    
    def get_best_k(self, data):
        # The maximum clusters is based on a heuristic to reduce computation time
        # significantly when there are a large number of products under a particular
        # group. The intent is to reduce the number of allowed clusters to a 
        # reasonable number.
        max_clusters = int(np.min([self.log(data.shape[0], 1.1), data.shape[0]])) + 1
        best_scores = np.zeros(len(range(2, max_clusters)))
        
        for i in range(2, max_clusters):
            
            clusters,centres,cluster_df = self.k_pod(data, i)
            
            # The following two conditions account for when the number of clusters assignable is just 1
            # which is empirically deemed unacceptable if the number of samples is more than 5, but
            # deemed acceptable if the number of samples is less than or equal to five.
            if np.unique(clusters).shape[0] == 1 & np.unique(clusters).shape[0] > 5:
                best_scores[i-2] = np.nan
            elif np.unique(clusters).shape[0] == 1 & np.unique(clusters).shape[0] <= 5:
                best_scores[i-2] = 0.9
            
            # scikit-learn requires that 1 < n_clusters < n_samples. That is, that the valid number of
            # clusters must be between 2 and the number of samples - 1. This condition as coded fails
            # when the number of clusters is 2 and the number of samples is also 2. The code below fixes
            # that issues manually for the silouette score by assigning a value 1 (evaluate the formula
            # to see that the result will be 1 for such a case).
            elif np.unique(clusters).shape[0] == clusters.shape[0]:
                best_scores[i-2] = 1
            else:
                best_scores[i-2] = silhouette_score(cluster_df, clusters)
            
            # Finally, if the number of clusters assignable becomes 1 after a certain point, then it is
            # safe to ignore all additional values and assign them a constant score:
            if (np.unique(clusters).shape[0] == 1) & (i > 5):
                best_scores[i-2:] = 0.9
                break
        
        # Good Indices = indices of best_score that are 1 std. deviation below the maximum (to avoid overfitting)
        fraction=1
        best_index=-1
        
        if max_clusters<3:
            
            while best_index == -1:
                
                stdv_indices = best_scores < (best_scores.max() - (fraction * np.std(best_scores)))
                pos_gradient = np.gradient(best_scores, axis=0) > 0
                good_indices = np.logical_and(stdv_indices, pos_gradient)
                
                # Best Index  = the highest index that has the highest score and a positive gradient
                if best_scores[good_indices].shape[0] == 0:
                    if fraction < 0.0025:
                        best_index =  np.argmax(best_scores)
                    fraction   /= 2
                else:
                    best_index = len(good_indices) - np.argmax(good_indices[::-1]) - 1
        else:
            best_index=np.argmax(best_scores)
        
        return (best_index+2, best_scores, times_taken, list(range(2, max_clusters)))
