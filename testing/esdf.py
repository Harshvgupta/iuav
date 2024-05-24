# import open3d as o3d
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R

# # Define paths
# left_depth_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\depth_left'  # Folder containing left .npy depth images
# right_depth_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\depth_right'  # Folder containing right .npy depth images
# left_pose_file = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\pose_left.txt'
# right_pose_file = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\pose_right.txt'
# left_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\image_left'  # Folder containing left images
# right_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\image_right'  # Folder containing right images

# # Load depth images
# left_depth_image_files = sorted([f for f in os.listdir(left_depth_images_path) if f.endswith('.npy')])
# right_depth_image_files = sorted([f for f in os.listdir(right_depth_images_path) if f.endswith('.npy')])

# left_depth_images = [np.load(os.path.join(left_depth_images_path, f)) for f in left_depth_image_files]
# right_depth_images = [np.load(os.path.join(right_depth_images_path, f)) for f in right_depth_image_files]

# # Load RGB images
# left_image_files = sorted([f for f in os.listdir(left_images_path) if f.endswith('.png')])
# right_image_files = sorted([f for f in os.listdir(right_images_path) if f.endswith('.png')])

# left_images = [o3d.io.read_image(os.path.join(left_images_path, f)) for f in left_image_files]
# right_images = [o3d.io.read_image(os.path.join(right_images_path, f)) for f in right_image_files]

# # Load camera poses
# def load_poses(pose_file):
#     poses = []
#     with open(pose_file, 'r') as file:
#         for line in file:
#             elements = line.strip().split()
#             if len(elements) == 7:
#                 position = np.array(elements[:3], dtype=float)
#                 quaternion = np.array(elements[3:], dtype=float)
#                 rotation = R.from_quat(quaternion).as_matrix()
#                 pose = np.eye(4)
#                 pose[:3, :3] = rotation
#                 pose[:3, 3] = position
#                 poses.append(pose)
#             else:
#                 print(f"Skipping invalid pose line: {line}")
#     return poses

# left_poses = load_poses(left_pose_file)
# right_poses = load_poses(right_pose_file)

# # TSDF volume parameters
# voxel_size = 0.05
# truncation_distance = 2.0 * voxel_size

# tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
#     voxel_length=voxel_size,
#     sdf_trunc=truncation_distance,
#     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
# )

# # Camera intrinsic parameters (adjust as needed)
# intrinsic = o3d.camera.PinholeCameraIntrinsic(
#     width=640,
#     height=480,
#     fx=525.0,
#     fy=525.0,
#     cx=319.5,
#     cy=239.5
# )

# # Integrate depth images into TSDF volume
# for left_depth, right_depth, left_image, right_image, left_pose, right_pose in zip(
#         left_depth_images, right_depth_images, left_images, right_images, left_poses, right_poses):
    
#     left_depth_o3d = o3d.geometry.Image(left_depth)
#     right_depth_o3d = o3d.geometry.Image(right_depth)

#     left_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         left_image, left_depth_o3d, convert_rgb_to_intensity=False)
#     right_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         right_image, right_depth_o3d, convert_rgb_to_intensity=False)

#     tsdf_volume.integrate(left_rgbd, intrinsic, left_pose)
#     tsdf_volume.integrate(right_rgbd, intrinsic, right_pose)

# # Extract point cloud and mesh from TSDF volume
# point_cloud = tsdf_volume.extract_point_cloud()
# mesh = tsdf_volume.extract_triangle_mesh()
# mesh.compute_vertex_normals()

# # Visualize point cloud and mesh
# o3d.visualization.draw_geometries([point_cloud, mesh])

# # Get the voxel size and volume extent from the TSDF volume
# voxel_size = tsdf_volume.voxel_length
# volume_extent = tsdf_volume.get_volume_extent()

# # Calculate the dimensions of the ESDF grid
# grid_dimensions = np.ceil((volume_extent.max_bound - volume_extent.min_bound) / voxel_size).astype(int)

# # Initialize ESDF grid
# esdf = np.ones(grid_dimensions) * np.inf

# # Compute ESDF using distance transform
# vertices = np.asarray(mesh.vertices)
# distances = np.linalg.norm(vertices - volume_extent.min_bound, axis=1)
# valid_indices = (distances <= truncation_distance)

# if np.any(valid_indices):
#     vertices_grid = ((vertices[valid_indices] - volume_extent.min_bound) / voxel_size).astype(int)
#     esdf[vertices_grid[:, 0], vertices_grid[:, 1], vertices_grid[:, 2]] = distances[valid_indices]

#     esdf = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[valid_indices])).compute_nearest_neighbor_distance()

# # Visualize ESDF slice
# plt.imshow(esdf[:, :, esdf.shape[2] // 2])
# plt.title('ESDF Slice')
# plt.colorbar()
# plt.show()
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Define paths
left_depth_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\depth_left'  # Folder containing left .npy depth images
right_depth_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\depth_right'  # Folder containing right .npy depth images
left_pose_file = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\pose_left.txt'
right_pose_file = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\pose_right.txt'
left_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\image_left'  # Folder containing left images
right_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\image_right'  # Folder containing right images
flow_and_mask_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\flow'
left_segmentation_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\seg_left'
right_segmentation_images_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\iuav\iuav\data\P001\seg_right'


# # Load depth images
# left_depth_image_files = sorted([f for f in os.listdir(left_depth_images_path) if f.endswith('_left_depth.npy')])
# left_depth_images = [np.load(os.path.join(left_depth_images_path, f)) for f in left_depth_image_files]

# # Load poses
# def load_poses(pose_file):
#     poses = []
#     with open(pose_file, 'r') as file:
#         for line in file:
#             elements = line.strip().split()
#             if len(elements) == 7:  # Ensure the line has 7 elements
#                 position = np.array(elements[:3], dtype=float)
#                 quaternion = np.array(elements[3:], dtype=float)
#                 rotation = R.from_quat(quaternion).as_matrix()
#                 pose = np.eye(4)
#                 pose[:3, :3] = rotation
#                 pose[:3, 3] = position
#                 poses.append(pose)
#             else:
#                 print(f"Skipping invalid pose line: {line}")
#     return poses

# left_poses = load_poses(left_pose_file)

# # Ensure we have the same number of images and poses for left
# assert len(left_depth_images) == len(left_poses), "Number of left depth images and left poses must be the same"

# # Camera intrinsic parameters (use correct values)
# fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0

# intrinsic = o3d.camera.PinholeCameraIntrinsic(
#     width=left_depth_images[0].shape[1],
#     height=left_depth_images[0].shape[0],
#     fx=fx, fy=fy,
#     cx=cx, cy=cy
# )

# # TSDF volume parameters
# voxel_size = 0.05
# truncation_distance = 2.0 * voxel_size

# tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
#     voxel_length=voxel_size,
#     sdf_trunc=truncation_distance,
#     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
# )

# # Convert depth images to point clouds and integrate them into the TSDF volume
# for idx, (left_depth_image, left_pose) in enumerate(zip(left_depth_images, left_poses)):
#     # Convert depth image to point cloud
#     depth_image = o3d.geometry.Image(left_depth_image.astype(np.float32))
#     rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         o3d.geometry.Image(np.zeros((left_depth_image.shape[0], left_depth_image.shape[1], 3), dtype=np.uint8)),
#         depth_image,
#         convert_rgb_to_intensity=False
#     )

#     point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
#         rgbd_image,
#         intrinsic,
#         extrinsic=np.linalg.inv(left_pose)
#     )
    
#     # Integrate point cloud into TSDF volume
#     tsdf_volume.integrate(rgbd_image, intrinsic, np.linalg.inv(left_pose))

# # Extract a point cloud from the TSDF volume
# integrated_point_cloud = tsdf_volume.extract_point_cloud()

# # Print point cloud for debugging
# print("Integrated Point Cloud:")
# print(integrated_point_cloud)

# # Extract the mesh for visualization
# mesh = tsdf_volume.extract_triangle_mesh()
# mesh.compute_vertex_normals()

# # Visualize the point cloud and mesh
# o3d.visualization.draw_geometries([integrated_point_cloud, mesh])

# # Extract voxel grid from the TSDF volume
# voxel_grid = tsdf_volume.extract_voxel_point_cloud()  # Correct method
# voxels = voxel_grid.get_voxels()

# # Initialize ESDF grid
# sdf = np.array([v.sdf for v in voxels])
# grid_size = voxel_grid.get_max_bound() - voxel_grid.get_min_bound()
# esdf = np.ones(grid_size.astype(int)) * float('inf')

# # Update ESDF with distance to the nearest surface
# for i in range(int(grid_size[0])):
#     for j in range(int(grid_size[1])):
#         for k in range(int(grid_size[2])):
#             if sdf[i, j, k] < truncation_distance:
#                 esdf[i, j, k] = np.linalg.norm([i, j, k] - np.array([i, j, k]))

# # Visualization of ESDF slices
# plt.imshow(esdf[:, :, esdf.shape[2] // 2])
# plt.title('ESDF Slice')
# plt.colorbar()
# plt.show()
# Function to read numpy files


# Function to read numpy files
def read_numpy_file(file_path):
    return np.load(file_path)

# Load all depth images
left_depth_image_files = sorted([f for f in os.listdir(left_depth_images_path) if f.endswith('_left_depth.npy')])
left_depth_images = [read_numpy_file(os.path.join(left_depth_images_path, f)) for f in left_depth_image_files]

right_depth_image_files = sorted([f for f in os.listdir(right_depth_images_path) if f.endswith('_right_depth.npy')])
right_depth_images = [read_numpy_file(os.path.join(right_depth_images_path, f)) for f in right_depth_image_files]

# Load poses
def load_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as file:
        for line in file:
            elements = line.strip().split()
            if len(elements) == 7:
                position = np.array(elements[:3], dtype=float)
                quaternion = np.array(elements[3:], dtype=float)
                rotation = R.from_quat(quaternion).as_matrix()
                pose = np.eye(4)
                pose[:3, :3] = rotation
                pose[:3, 3] = position
                poses.append(pose)
            else:
                print(f"Skipping invalid pose line: {line}")
    return poses

left_poses = load_poses(left_pose_file)
right_poses = load_poses(right_pose_file)

# Ensure we have the same number of images and poses for both left and right
assert len(left_depth_images) == len(left_poses), "Number of left depth images and left poses must be the same"
assert len(right_depth_images) == len(right_poses), "Number of right depth images and right poses must be the same"

# Camera intrinsic parameters
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=left_depth_images[0].shape[1],
    height=left_depth_images[0].shape[0],
    fx=320.0, fy=320.0,
    cx=320.0, cy=240.0
)

# Create TSDF volume
voxel_size = 0.05
tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size,
    sdf_trunc=voxel_size * 5,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

# Function to integrate point cloud into TSDF volume
def integrate_depth_image(tsdf_volume, depth_image, intrinsic, pose):
    depth_image_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
    color_image = o3d.geometry.Image(np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image_o3d, convert_rgb_to_intensity=False)
    tsdf_volume.integrate(rgbd_image, intrinsic, np.linalg.inv(pose))

# Integrate all depth images (left and right) into the TSDF volume
for depth_image, pose in zip(left_depth_images, left_poses):
    integrate_depth_image(tsdf_volume, depth_image, intrinsic, pose)

for depth_image, pose in zip(right_depth_images, right_poses):
    integrate_depth_image(tsdf_volume, depth_image, intrinsic, pose)

# Extract and visualize the integrated point cloud
integrated_point_cloud = tsdf_volume.extract_point_cloud()
print("Integrated Point Cloud:")
print(integrated_point_cloud)
o3d.visualization.draw_geometries([integrated_point_cloud])

# Extract and visualize the mesh
mesh = tsdf_volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

# Extract voxel grid from the TSDF volume
voxel_grid = tsdf_volume.extract_voxel_point_cloud()
voxels = voxel_grid.get_voxels()

# Initialize ESDF grid
sdf = np.array([v.sdf for v in voxels])
grid_size = voxel_grid.get_max_bound() - voxel_grid.get_min_bound()
esdf = np.ones(grid_size.astype(int)) * float('inf')

# Update ESDF with distance to the nearest surface
for i in range(int(grid_size[0])):
    for j in range(int(grid_size[1])):
        for k in range(int(grid_size[2])):
            if sdf[i, j, k] < truncation_distance:
                esdf[i, j, k] = np.linalg.norm([i, j, k] - np.array([i, j, k]))

# Visualization of ESDF slices
plt.imshow(esdf[:, :, esdf.shape[2] // 2])
plt.title('ESDF Slice')
plt.colorbar()
plt.show()