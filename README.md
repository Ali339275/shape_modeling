# 3D Shape Modeling from Silhouettes

## Overview

This project investigates 3D shape reconstruction from 2D binary silhouette images using two complementary approaches:
	1. **Voxel-based visual hull reconstruction
	2. **Neural implicit representation using a multilayer perceptron (MLP)

Given multiple silhouettes of an object captured from different viewpoints, the goal is to reconstruct a 3D shape that is consistent with all observations. The reconstructed shape corresponds to the visual hull, defined as the maximal volume compatible with the provided silhouettes.

The project uses a 3D model of Al as reference and a set of 12 silhouette projections with known camera calibration parameters.

---

## Data and Project Structure

### The repository contains:
	•	al.off — reference 3D mesh of the object
	•	images/ — 12 silhouette images captured from different viewpoints
	•	Camera calibration matrices mapping 3D points to image coordinates
	•	Python scripts for voxel carving, neural implicit learning, and visualization
	•	output/ — generated meshes and intermediate data

Before running the code, activate the Python environment:

source /opt/python-ensimag/bin/activate

To visualize the reference mesh:

python show_mesh.py al.off

Visual Hull Reconstruction (Voxel Carving)

## Method

The first part reconstructs the visual hull using a voxel carving approach.

A regular 3D grid is defined over the object volume. Each voxel is projected into the silhouette images using the camera calibration matrices. Voxels that project outside at least one silhouette are removed. The remaining voxels represent the intersection of all visual cones defined by the silhouettes.

### Implementation Details
	•	Grid resolution is configurable to balance accuracy and computation time
	•	Projections are computed efficiently using NumPy vectorization
	•	The final voxel occupancy grid is converted into a surface mesh using the marching cubes algorithm
	•	Outputs include:
	•	alvoxels.off: voxel-based surface mesh
	•	occupancy.npy: numerical voxel occupancy grid

The resulting mesh can be visualized and compared with the reference model.

## Neural Implicit 3D Representation

### Method

The second part replaces the explicit voxel grid with a neural implicit representation learned by a multilayer perceptron (MLP).

The MLP learns an implicit occupancy function:

f(x, y, z) → {0, 1}

which predicts whether a 3D point lies inside or outside the object. The network is trained using occupancy labels derived from the visual hull.

This approach encodes the 3D shape compactly in the network parameters, rather than in an explicit voxel grid.

### Training Data
	•	Input: 3D coordinates (x, y, z)
	•	Output: binary occupancy (inside / outside)
	•	Training samples can be:
	•	regularly sampled grid points
	•	randomly sampled points in 3D space

Class imbalance between inside and outside points is handled through loss weighting or sampling strategies.


## Output

After training, the MLP is evaluated on a regular 3D grid. The predicted occupancy field is converted into a surface mesh:
	•	alimplicit.off: neural implicit surface reconstruction


## GPU Execution

Training the neural implicit model requires GPU acceleration.

Execution is performed on the Ensimag / Grenoble INP GPU cluster using Slurm:

ssh -YK nash.ensimag.fr

srun –gres=shard:1 –cpus-per-task=8 –mem=12GB –unbuffered python MLPimplicit3D.py


## Experiments and Extensions

### The project includes several experimental variations:
	•	Saving and reloading trained neural models
	•	Memory comparison between:
	•	voxel occupancy grids
	•	neural implicit representations
	•	silhouette images and calibration data
	•	Training with randomly sampled 3D points
	•	Reducing outside-point imbalance through selective sampling
	•	Modifying the MLP architecture:
	•	number of layers
	•	hidden layer size

These experiments explore trade-offs between reconstruction quality, memory efficiency, and model complexity.


## Comparison of Representations

###  The project enables qualitative and quantitative comparison between:  
	•	the original reference mesh
	•	the voxel-based visual hull
	•	the neural implicit reconstruction

Each representation illustrates different compromises between accuracy, smoothness, memory usage, and computational cost.
