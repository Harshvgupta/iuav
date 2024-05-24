import torch
import matplotlib.pyplot as plt
from minimum_snap import UAVTrajectoryPlanner


def demo_minimum_snap_simple():
    # waypoints = torch.tensor([[0, 0], [1, 2], [2, -1], [4, 8], [5, 2]], dtype=torch.float64).t()
    waypoints = torch.tensor([[0, 0,0], [1, 0,1], [1, 2,1], [0,2,1]], dtype=torch.float64).t()
    # waypoints = torch.tensor([[1, 0,0],[1.6,0.2,0], [2, 1,0],[1.6,1.8,0], [1, 2,0],[0.2,1.6,0], [0,1,0],[0.4,0.2,0],[1,0,0]], dtype=torch.float64).t()
    start_vel, start_acc, end_vel, end_acc = torch.tensor([0, 0,0]), torch.tensor([0, 0,0]), torch.tensor([0, 0,0]), torch.tensor([0, 0,0])
    total_time = 20.0
    poly_order = 5
    custom_time_stamps = [0.0, 8.0, 16.0, 24.0]

    planner = UAVTrajectoryPlanner(total_time, poly_order, start_vel, start_acc, end_vel, end_acc, dtype=torch.float64, device='cpu')
    polys_x, polys_y, polys_z, time_stamps = planner(waypoints,custom_time_stamps)
    print("Polynomials with custom time stamps:", polys_x, polys_y, polys_z)
    # polys_x, polys_y, polys_z, time_stamps = planner(waypoints)
    # print("Polynomials with arranged time stamps:", polys_x, polys_y, polys_z)

    # planner.plot_trajectory(waypoints, polys_x, polys_y,polys_z, time_stamps)
    planner.plot_trajectory_and_derivatives(waypoints, polys_x, polys_y,polys_z,  time_stamps)

demo_minimum_snap_simple()