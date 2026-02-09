import matplotlib.image as mpimg
import numpy as np
from skimage import measure
import trimesh


# Camera Calibration for Al's image[1..12].pgm   
calib = np.array([
    [-78.8596, -178.763, -127.597, 300, -230.924, 0, -33.6163, 300,
     -0.525731, 0, -0.85065, 2],
    [0, -221.578, 73.2053, 300, -178.763, -127.597, -78.8596, 300,
     0, -0.85065, -0.525731, 2],
    [ 78.8596, -178.763, -127.597, 300, -73.2053, 0, -221.578, 300,
     0.525731, 0, -0.85065, 2],
    [0, 33.6163, -230.924, 300, -178.763, 127.597, -78.8596, 300,
     0, 0.85065, -0.525731, 2],
    [-78.8596, -178.763, 127.597, 300, 73.2053, 0, 221.578, 300,
     -0.525731, 0, 0.85065, 2],
    [78.8596, -178.763, 127.597, 300, 230.924, 0, 33.6163, 300,
     0.525731, 0, 0.85065, 2],
    [0, -221.578, -73.2053, 300, 178.763, -127.597, 78.8596, 300,
     0, -0.85065, 0.525731, 2],
    [0, 33.6163, 230.924, 300, 178.763, 127.597, 78.8596, 300,
     0, 0.85065, 0.525731, 2],
    [-33.6163, -230.924, 0, 300, -127.597, -78.8596, 178.763, 300,
     -0.85065, -0.525731, 0, 2],
    [-221.578, -73.2053, 0, 300, -127.597, 78.8596, 178.763, 300,
     -0.85065, 0.525731, 0, 2],
    [221.578, -73.2053, 0, 300, 127.597, 78.8596, -178.763, 300,
     0.85065, 0.525731, 0, 2],
    [33.6163, -230.924, 0, 300, 127.597, -78.8596, -178.763, 300,
     0.85065, -0.525731, 0, 2]
])


# Build 3D grids
# 3D Grids are of size: resolution x resolution x resolution/2
resolution = 100
step = 2 / resolution

# Voxel coordinates
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

# Voxel occupancy
occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

# Voxels are initially occupied then carved with silhouette information
occupancy.fill(1)



# ---------- MAIN ----------
if __name__ == "__main__":
    
    i = 1
    # read the input silhouettes
    myFile = "image{0}.pgm".format(i)
    print(myFile)
    img = mpimg.imread(myFile)
    
    if img.dtype == np.float32:  # if not integer
        img = (img * 255).astype(np.uint8) #this make black(0) white(255)

    

    # #TODO: Compute grid projection in images
    # for ix in range(resolution):
    #     for iy in range(resolution):
    #         for iz in range(resolution//2):

    #             if occupancy[ix, iy, iz] == 0:
    #                 continue

    #             vx = X[ix, iy, iz]
    #             vy = Y[ix, iy, iz]
    #             vz = Z[ix, iy, iz]
    #             voxel_3D = np.array([vx, vy, vz, 1])

    #             # Check against all silhouettes
    #             for cam in range(12):
    #                 P = calib[cam].reshape(3,4)
    #                 proj = P @ voxel_3D

    #                 u = int(proj[0] / proj[2])
    #                 v = int(proj[1] / proj[2])

    #                 myFile = "image{0}.pgm".format(cam)
    #                 img = mpimg.imread(myFile) 
    #                 if img.dtype == np.float32:  # if not integer
    #                     img = (img * 255).astype(np.uint8)
                 
    #                 if u <0 or u>=img.shape[0] or v <0 or v>=img.shape[1]:
    #                     occupancy[ix,iy,iz] = 0
    #                     break     
    #                 if img[u,v] == 0:
    #                     occupancy[ix,iy,iz] = 0
    #                     break    

    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    Zf = Z.reshape(-1)
    ones = np.ones_like(Xf)
    voxel_3D = np.vstack([Xf, Yf, Zf, ones])  
    occ = occupancy.reshape(-1)   
    
    for cam in range(12):

        # Load projection matrix
        P = calib[cam].reshape(3, 4)

        # Project all voxels
        proj = P @ voxel_3D                  # (3, N)

        # Perspective division
        u = (proj[0, :] / proj[2, :]).astype(int)
        v = (proj[1, :] / proj[2, :]).astype(int)


        # Load silhouette image ONCE
        img = mpimg.imread(f"image{cam}.pgm")
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)

        H, W = img.shape

        # ---- Out-of-bounds mask ----
        out_of_bounds = (u < 0) | (u >= W) | (v < 0) | (v >= H)

        # ---- Silhouette test (only for valid projections) ----
        silhouette_fail = np.zeros_like(out_of_bounds, dtype=bool)
        valid = ~out_of_bounds

        silhouette_fail[valid] = (img[u[valid], v[valid]] == 0)

        # ---- Carve voxels ----
        occ[out_of_bounds | silhouette_fail] = 0

# ---- Restore 3D occupancy grid ----
    occupancy = occ.reshape(resolution, resolution, resolution // 2)


    # TODO: Update grid occupancy

    # Voxel visualization

    # Use the marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(occupancy, 0.25)

    # Export in a standard file format
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alvoxels.off')
 
