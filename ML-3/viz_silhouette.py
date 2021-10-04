# code for Visualizing silhouette_score and clusters

def viz(data, k_cluster):

	
	import numpy as np
	import seaborn as sns
	import matplotlib.cm as cm
	from matplotlib import pyplot as plt

	n_clusters = k_cluster
	X = np.array(data)
	for K in n_clusters:
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.set_size_inches(18, 7)
		model = KMeans(n_clusters = K, random_state = 10)
		cluster_labels = model.fit_predict(X)
		silhouette_avg = silhouette_score(X, cluster_labels)
		sample_silhouette_values = silhouette_samples(X, cluster_labels)
		y_lower = 10
		for i in range(K):
		    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
		    ith_cluster_silhouette_values.sort()
		    size_cluster_i = ith_cluster_silhouette_values.shape[0]
		    y_upper = y_lower + size_cluster_i
		    color = cm.nipy_spectral(float(i) / K)
		    ax1.fill_betweenx(np.arange(y_lower, y_upper),
		                      0, ith_cluster_silhouette_values,
		                      facecolor=color, edgecolor=color, alpha=0.7)
		    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
		    y_lower = y_upper + 10 
		ax1.set_title("Silhouette Plot")
		ax1.set_xlabel("Silhouette coefficient")
		ax1.set_ylabel("Cluster label")
		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
		ax1.set_yticks([])  
		ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8])
		colors = cm.nipy_spectral(cluster_labels.astype(float) / K)
		ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
		centers = model.cluster_centers_    
		for i, c in enumerate(centers):
		    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')
		ax2.set_title("Clusters")
		ax2.set_xlabel("Spending Score")
		ax2.set_ylabel("Annual Income")    
		plt.suptitle(("Silhouette Analysis for K-Means Clustering with n_clusters = %d" % K), fontsize=14, 
		             fontweight='bold')
	plt.show()
	
	

