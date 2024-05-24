import torch
from torch import nn
import qpth
from qpth.qp import QPFunction
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class UAVTrajectoryPlanner(nn.Module):
    def __init__(self,waypoints, total_time, poly_order, start_vel, start_acc, end_vel, end_acc, dtype=torch.float64, device='cpu'):
        super(UAVTrajectoryPlanner, self).__init__()
        self.waypoints = waypoints
        self.total_time = total_time
        self.poly_order = poly_order
        self.start_vel = start_vel
        self.start_acc = start_acc
        self.end_vel = end_vel
        self.end_acc = end_acc
        self.dtype = dtype
        self.device = device

    def arrange_time(self, waypoints, total_time):
        """
        Arrange waypoints in time.
        
        Args:
            waypoints (Tensor): waypoints tensor
            total_time (float): total time for the trajectory
            
        Returns:
            Tensor: arranged time tensor
        """
        differences = waypoints[:, 1:] - waypoints[:, :-1]
        distances = torch.sqrt(torch.sum(differences ** 2, dim=0))
        time_fraction = total_time / torch.sum(distances)
        arranged_time = torch.cat([torch.tensor([0]), torch.cumsum(distances * time_fraction, dim=0)])
        print(arranged_time)
        return arranged_time
    def calculate_time_vector(self,time, order, derivative_order):
        """
        Calculate time vector for a given time, polynomial order, and derivative order.
        
        Args:
            time (float): the time at which the vector is calculated
            order (int): order of the polynomial
            derivative_order (int): derivative order
            
        Returns:
            Tensor: calculated time vector
        """
        time_vector = torch.zeros(order + 1)
        for i in range(derivative_order + 1, order + 2):
            if i - derivative_order - 1 > 0:
                product = torch.prod(torch.arange(i - derivative_order, i, dtype=torch.float32))
            else:
                product = torch.prod(torch.arange(i - derivative_order, i, dtype=torch.float32))
            time_vector[i - 1] = product * (time ** (i - derivative_order - 1))
        return time_vector
    def compute_Q_matrix(self,poly_order, derivative_order, start_time, end_time):
        """
        Compute the Q matrix for minimum snap problem.
        
        Args:
            poly_order (int): order of the polynomial
            derivative_order (int): derivative order
            start_time (float): start time
            end_time (float): end time
            
        Returns:
            Tensor: Q matrix
        """
        time_diff_powers = torch.zeros((poly_order - derivative_order) * 2 + 1)
        for i in range((poly_order - derivative_order) * 2 + 1):
            time_diff_powers[i] = end_time ** (i + 1) - start_time ** (i + 1)

        Q_matrix = torch.zeros(poly_order + 1, poly_order + 1)
        for i in range(derivative_order + 1, poly_order + 2):
            for j in range(i, poly_order + 2):
                k1 = i - derivative_order - 1
                k2 = j - derivative_order - 1
                k = k1 + k2 + 1
                prod_k1 = torch.prod(torch.tensor(range(k1 + 1, k1 + derivative_order + 1)))
                prod_k2 = torch.prod(torch.tensor(range(k2 + 1, k2 + derivative_order + 1))) 
                Q_matrix[i - 1, j - 1] = prod_k1 * prod_k2 / k * time_diff_powers[k - 1]
                Q_matrix[j - 1, i - 1] = Q_matrix[i - 1, j - 1]

        return Q_matrix

    def solve_minimum_snap(self, waypoints, time_stamps, poly_order, start_vel, start_acc, end_vel, end_acc):
        """
        Solve the minimum snap problem for a single axis.
        
        Args:
            waypoints (Tensor): waypoints
            time_stamps (Tensor): time stamps for each segment
            poly_order (int): polynomial order
            start_vel (float): start velocity
            start_acc (float): start acceleration
            end_vel (float): end velocity
            end_acc (float): end acceleration
            dtype (data type): data type for tensors
            device (str): computation device ('cpu' or 'cuda')
            
        Returns:
            Tensor: coefficients of the polynomial
        """
        print(waypoints,time_stamps[0])
        start_pos = waypoints[0]
        end_pos = waypoints[-1]
        num_segments = len(waypoints) - 1
        num_coefficients = poly_order + 1

        # Compute Q matrix for all segments
        Q_all = torch.block_diag(*[self.compute_Q_matrix(poly_order, 3, time_stamps[i], time_stamps[i + 1]) for i in range(num_segments)])
        b_all = torch.zeros(Q_all.shape[0])

        # Setup equality constraints
        Aeq = torch.zeros(4 * num_segments + 2, num_coefficients * num_segments)
        beq = torch.zeros(4 * num_segments + 2)
        Aeq[0:3, :num_coefficients] = torch.stack([
            self.calculate_time_vector(time_stamps[0], poly_order, 0),
            self.calculate_time_vector(time_stamps[0], poly_order, 1),
            self.calculate_time_vector(time_stamps[0], poly_order, 2)])
        Aeq[3:6, -num_coefficients:] = torch.stack([
            self.calculate_time_vector(time_stamps[-1], poly_order, 0),
            self.calculate_time_vector(time_stamps[-1], poly_order, 1),
            self.calculate_time_vector(time_stamps[-1], poly_order, 2)])
        beq[0:6] = torch.tensor([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc])

        # Middle waypoints constraints
        num_eq_constraints = 6
        for i in range(1, num_segments):
            Aeq[num_eq_constraints, i * num_coefficients:(i + 1) * num_coefficients] = self.calculate_time_vector(time_stamps[i], poly_order, 0)
            beq[num_eq_constraints] = waypoints[i]
            num_eq_constraints += 1

        # Continuity constraints
        for i in range(1, num_segments):
            time_vector_p = self.calculate_time_vector(time_stamps[i], poly_order, 0)
            time_vector_v = self.calculate_time_vector(time_stamps[i], poly_order, 1)
            time_vector_a = self.calculate_time_vector(time_stamps[i], poly_order, 2)
            Aeq[num_eq_constraints:num_eq_constraints + 3, (i - 1) * num_coefficients:(i + 1) * num_coefficients] = torch.stack([
                torch.cat([time_vector_p, -time_vector_p]),
                torch.cat([time_vector_v, -time_vector_v]),
                torch.cat([time_vector_a, -time_vector_a])])
            num_eq_constraints += 3

        # Convert to the specified data type and device
        G_dummy = torch.zeros(1, Q_all.size(0), Q_all.size(0), dtype=torch.float64)
        h_dummy = torch.zeros(1, Q_all.size(0), dtype=torch.float64)
        # print(Q_all)
        Q_all += torch.eye(Q_all.size(0), dtype=torch.float64) * 1e-6
        Q_all = Q_all.to(dtype=self.dtype, device=self.device)
        b_all = b_all.to(dtype=self.dtype, device=self.device)
        Aeq = Aeq.to(dtype=self.dtype, device=self.device)
        beq = beq.to(dtype=self.dtype, device=self.device)
        G_dummy = G_dummy.to(dtype=self.dtype, device=self.device)
        h_dummy = h_dummy.to(dtype=self.dtype, device=self.device)

        # Solve the QP problem
        solver_options = {'eps':1e-24, 'maxIter': 100, 'solver': qpth.qp.QPSolvers.PDIPM_BATCHED}
        solution = QPFunction(verbose=1, **solver_options)(Q_all, b_all, G_dummy, h_dummy, Aeq, beq)
        polynomial_coefficients = solution.view(num_segments, num_coefficients).transpose(0, 1)
        return polynomial_coefficients
    
    def evaluate_polynomial(self,polynomial_coefficients, time, derivative_order):
        """
        Evaluate a polynomial at a given time.
        
        Args:
            polynomial_coefficients (Tensor): coefficients of the polynomial
            time (float): time at which to evaluate the polynomial
            derivative_order (int): derivative order
            
        Returns:
            float: value of the polynomial at the given time
        """
        value = 0
        polynomial_order = len(polynomial_coefficients) - 1
        if derivative_order <= 0:
            for i in range(polynomial_order + 1):
                value += polynomial_coefficients[i] * time ** i
        else:
            for i in range(derivative_order, polynomial_order + 1):
                value += polynomial_coefficients[i] * np.prod(range(i - derivative_order + 1, i + 1)) * time ** (i - derivative_order)
        return value

    def evaluate_polynomials(self,polynomial_coefficients, time_stamps, times, derivative_order):
        """
        Evaluate polynomials over a time range.
        
        Args:
            polynomial_coefficients (Tensor): coefficients of the polynomials
            time_stamps (Tensor): time stamps for each segment
            times (Tensor): times at which to evaluate the polynomials
            derivative_order (int): derivative order
            
        Returns:
            Tensor: values of the polynomials at the given times
        """
        num_points = times.size(0)
        values = torch.zeros(num_points)
        index = 0
        for i in range(num_points):
            time = times[i]
            if time < time_stamps[index]:
                values[i] = 0
            else:
                while index < len(time_stamps) - 1 and time > time_stamps[index + 1] + 0.0001:
                    index += 1
                values[i] = self.evaluate_polynomial(polynomial_coefficients[:, index], time, derivative_order)
        return values

    def plot_trajectory1(self,waypoints, polys_x, polys_y,polys_z ,time_stamps):

        """
        Plot the minimum snap trajectory.
        
        Args:
            waypoints (Tensor): waypoints
            polys_x (Tensor): polynomial coefficients for x axis
            polys_y (Tensor): polynomial coefficients for y axis
            time_stamps (Tensor): time stamps for each segment
        """
        plt.plot(waypoints[0], waypoints[1],waypoints[2], '*r')
        plt.plot(waypoints[0], waypoints[1],waypoints[2], 'b--')
        plt.title('Minimum Snap Trajectory')
        colors = ['g', 'r', 'c']
        for i in range(polys_x.shape[1]):
            times = torch.arange(time_stamps[i], time_stamps[i+1], 0.01)
            x_values = self.evaluate_polynomials(polys_x, time_stamps, times, 0)
            y_values = self.evaluate_polynomials(polys_y, time_stamps, times, 0)
            z_values = self.evaluate_polynomials(polys_z, time_stamps, times, 0)
            plt.plot(x_values.detach().numpy(), y_values.detach().numpy(), z_values.detach().numpy(),colors[i % 3])
        plt.show()
    def plot_trajectory(self, waypoints, polys_x, polys_y, polys_z, time_stamps):
        """
        Plot the minimum snap trajectory in 3D.
        
        Args:
            waypoints (Tensor): waypoints in 3D space
            polys_x (Tensor): polynomial coefficients for x axis
            polys_y (Tensor): polynomial coefficients for y axis
            polys_z (Tensor): polynomial coefficients for z axis
            time_stamps (Tensor): time stamps for each segment
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot waypoints
        # ax.scatter(waypoints[0], waypoints[1], waypoints[2], color='r', marker='*')
        print("Waypoints:\n", waypoints)

        ax.scatter(waypoints[0, :].numpy(), waypoints[1, :].numpy(), waypoints[2, :].numpy(), color='r', marker='*')


        # Plot trajectory in segments
        colors = ['g', 'r', 'c']
        for i in range(polys_x.shape[1]):
            times = torch.arange(time_stamps[i], time_stamps[i + 1], 0.01)
            x_values = self.evaluate_polynomials(polys_x, time_stamps, times, 0)
            y_values = self.evaluate_polynomials(polys_y, time_stamps, times, 0)
            z_values = self.evaluate_polynomials(polys_z, time_stamps, times, 0)
            ax.plot(x_values.detach().numpy(), y_values.detach().numpy(), z_values.detach().numpy(), color=colors[i % 3])

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.title('Minimum Snap Trajectory')
        plt.show()
    def plot_trajectory_and_derivatives(self, waypoints, polys_x, polys_y,polys_z ,time_stamps):
        # Plot trajectory
        self.plot_trajectory(waypoints, polys_x, polys_y,polys_z , time_stamps)
        fig, axs = plt.subplots(4, 3, figsize=(12, 16))  # Prepare subplots in advance

        # Initialize arrays to hold all values for plotting (concatenate results from each segment)
        all_times = []
        all_x_positions = []
        all_y_positions = []
        all_z_positions = []
        all_x_velocities = []
        all_y_velocities = []
        all_z_velocities = []
        all_x_accelerations = []
        all_y_accelerations = []
        all_z_accelerations = []
        all_x_jerks = []
        all_y_jerks = []
        all_z_jerks = []

        for i in range(polys_x.shape[1]):
            times = torch.arange(time_stamps[i], time_stamps[i+1], 0.01)
            all_times.extend(times.numpy())  # Store times for overall plotting

            # Evaluate derivatives and store results
            all_x_positions.extend(self.evaluate_polynomials(polys_x, time_stamps, times, 0).detach().numpy())
            all_y_positions.extend(self.evaluate_polynomials(polys_y, time_stamps, times, 0).detach().numpy())
            all_z_positions.extend(self.evaluate_polynomials(polys_z, time_stamps, times, 0).detach().numpy())
            all_x_velocities.extend(self.evaluate_polynomials(polys_x, time_stamps, times, 1).detach().numpy())
            all_y_velocities.extend(self.evaluate_polynomials(polys_y, time_stamps, times, 1).detach().numpy())
            all_z_velocities.extend(self.evaluate_polynomials(polys_z, time_stamps, times, 1).detach().numpy())
            all_x_accelerations.extend(self.evaluate_polynomials(polys_x, time_stamps, times, 2).detach().numpy())
            all_y_accelerations.extend(self.evaluate_polynomials(polys_y, time_stamps, times, 2).detach().numpy())
            all_z_accelerations.extend(self.evaluate_polynomials(polys_z, time_stamps, times, 2).detach().numpy())
            all_x_jerks.extend(self.evaluate_polynomials(polys_x, time_stamps, times, 3).detach().numpy())
            all_y_jerks.extend(self.evaluate_polynomials(polys_y, time_stamps, times, 3).detach().numpy())
            all_z_jerks.extend(self.evaluate_polynomials(polys_z, time_stamps, times, 3).detach().numpy())

        # Now plot the collected data
        axs[0, 0].plot(all_times, all_x_positions)
        axs[0, 1].plot(all_times, all_y_positions)
        axs[0, 2].plot(all_times, all_z_positions)

        axs[1, 0].plot(all_times, all_x_velocities)
        axs[1, 1].plot(all_times, all_y_velocities)
        axs[1, 2].plot(all_times, all_z_velocities)

        axs[2, 0].plot(all_times, all_x_accelerations)
        axs[2, 1].plot(all_times, all_y_accelerations)
        axs[2, 2].plot(all_times, all_z_accelerations)

        axs[3, 0].plot(all_times, all_x_jerks)
        axs[3, 1].plot(all_times, all_y_jerks)
        axs[3, 2].plot(all_times, all_z_jerks)

        for ax in axs.flat:
            ax.set(xlabel='Time (s)', ylabel='Value')
            ax.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()


    def forward(self, waypoints,time_stamps=None):
        if time_stamps is None:
            time_stamps = self.arrange_time(waypoints, self.total_time)
        else:
            time_stamps = torch.tensor(time_stamps, dtype=torch.float64)
        
        print("time stamp is: ", time_stamps)
        
        if torch.cuda.is_available():
            polys_x = self.solve_minimum_snap(waypoints[0], time_stamps, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
            polys_y = self.solve_minimum_snap(waypoints[1], time_stamps, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
            polys_z = self.solve_minimum_snap(waypoints[2], time_stamps, self.poly_order, self.start_vel[2], self.start_acc[2], self.end_vel[2], self.end_acc[2])
        else:
            polys_x = self.solve_minimum_snap(waypoints[0], time_stamps, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
            polys_y = self.solve_minimum_snap(waypoints[1], time_stamps, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
            polys_z = self.solve_minimum_snap(waypoints[2], time_stamps, self.poly_order, self.start_vel[2], self.start_acc[2], self.end_vel[2], self.end_acc[2])

        return polys_x, polys_y, polys_z, time_stamps
