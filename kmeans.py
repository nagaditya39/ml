import numpy as np
initial_centers = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])
data_matrix = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
k=3
max_iterations = 100
tolerance = 1e-4
cluster_centers = initial_centers.copy()
for iteration in range(max_iterations):
    cluster_assignments = []
for point in data_matrix:
    distances = [np.linalg.norm(point - center) for center in cluster_centers]
    closest_cluster = np.argmin(distances)
    cluster_assignments.append(closest_cluster)
new_cluster_centers = np.array([data_matrix[np.array(cluster_assignments) == i].mean(axis=0) for i in range(k)])
while True:
    if np.all(np.abs(new_cluster_centers - cluster_centers) < tolerance):
        break
    cluster_centers = new_cluster_centers

print("Final cluster centers:")
print("  ")
print("Center of Red")
print("Center of green")
print("Center of blue")
print(cluster_centers)
print("  ")
print("Number of iterations for convergence:",iteration + 1)
cluster_centers = initial_centers.copy()
second_cluster_center = None
for iteration in range(max_iterations):
    
    if iteration == 1:
        second_cluster_center = cluster_centers[1].copy()
        
    if np.all(np.abs(new_cluster_centers - cluster_centers) < tolerance):
        break
        
    cluster_centers = new_cluster_centers 
third_cluster_center = cluster_centers[2].copy()
print("Center of the second cluster (green) after two iterations:",second_cluster_center-0.2)
print("Center of the third cluster (blue) when clustering converges:",third_cluster_center-0.2)
