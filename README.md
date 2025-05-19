# postprocess_MPI-Lag
A toolbox for postprocess MPI-Lag to analyze trajectories for TCG

- Calculate Dynamic Time Warping distance matrix for each trajectories (in parallel)
- Using Connected Component Labeling (CCL) to separate each vortex together.
- Hierarchical Clustering for trajectories using DTW distance matrix.
- Find all available points mapping to "class-median trajectories" by Dynamic Time Warping Paths.
