import torch
from torch import nn
import qpth
from qpth.qp import QPFunction
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(precision=10)

class TrajectoryUtils:
    @staticmethod
    def calculate_time_vector(time, order, derivative_order):
        time_vector = torch.zeros(order + 1)
        for i in range(derivative_order + 1, order + 2):
            if i - derivative_order - 1 > 0:
                product = torch.prod(torch.arange(i - derivative_order, i, dtype=torch.float32))
            else:
                product = torch.prod(torch.arange(i - derivative_order, i, dtype=torch.float32))
            time_vector[i - 1] = product * (time ** (i - derivative_order - 1))
        return time_vector

    @staticmethod
    def compute_Q_matrix(poly_order, derivative_order, start_time, end_time):
        time_diff_powers = torch.zeros((poly_order - derivative_order) * 2 + 1, dtype=torch.float64)
        for i in range((poly_order - derivative_order) * 2 + 1):
            time_diff_powers[i] = end_time ** (i + 1) - start_time ** (i + 1)

        Q_matrix = torch.zeros(poly_order + 1, poly_order + 1, dtype=torch.float64)
        for i in range(derivative_order + 1, poly_order + 2):
            for j in range(i, poly_order + 2):
                k1 = i - derivative_order - 1
                k2 = j - derivative_order - 1
                k = k1 + k2 + 1
                prod_k1 = torch.prod(torch.tensor(range(k1 + 1, k1 + derivative_order + 1), dtype=torch.float64))
                prod_k2 = torch.prod(torch.tensor(range(k2 + 1, k2 + derivative_order + 1), dtype=torch.float64))
                Q_matrix[i - 1, j - 1] = prod_k1 * prod_k2 / k * time_diff_powers[k - 1]
                Q_matrix[j - 1, i - 1] = Q_matrix[i - 1, j - 1]
        return Q_matrix

    @staticmethod
    def evaluate_polynomial(polynomial_coefficients, time, derivative_order):
        value = 0
        polynomial_order = len(polynomial_coefficients) - 1
        if derivative_order <= 0:
            for i in range(polynomial_order + 1):
                value += polynomial_coefficients[i] * time ** i
        else:
            for i in range(derivative_order, polynomial_order + 1):
                value += polynomial_coefficients[i] * np.prod(range(i - derivative_order + 1, i + 1)) * time ** (i - derivative_order)
        return value

    @staticmethod
    def evaluate_polynomials(polynomial_coefficients, time_stamps, times, derivative_order):
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
                values[i] = TrajectoryUtils.evaluate_polynomial(polynomial_coefficients[:, index], time, derivative_order)
        return values

class TrajectoryPlotter:
    @staticmethod
    def plot_trajectory(waypoints, polys_x, polys_y, time_stamps, iteration=None, clear_plot=False, flag=False):
        if clear_plot:
            plt.clf()
        plt.plot(waypoints[0], waypoints[1], '*r')
        plt.plot(waypoints[0], waypoints[1], 'b--')
        plt.title('Minimum Snap Trajectory')
        plt.xlim([-0.5, 6.0])
        plt.ylim([-0.5, 10.0])
        colors = ['g', 'r', 'c', 'm', 'y', 'k']
        for i in range(polys_x.shape[1]):
            times = torch.arange(time_stamps[i].item(), time_stamps[i+1].item(), 0.01, device=polys_x.device, dtype=polys_x.dtype)
            x_values = TrajectoryUtils.evaluate_polynomials(polys_x, time_stamps, times, 0)
            y_values = TrajectoryUtils.evaluate_polynomials(polys_y, time_stamps, times, 0)
            plt.plot(x_values.detach().numpy(), y_values.detach().numpy(), colors[i % len(colors)])
        if iteration is not None:
            plt.text(waypoints[0, -1], waypoints[1, -1], f"Iteration {iteration}", fontsize=12)
        if flag:
            plt.show()

class UAVTrajectoryPlanner(nn.Module):
    def __init__(self, waypoints, total_time, poly_order, start_vel, start_acc, end_vel, end_acc, dtype=torch.float32, device='cpu'):
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

    def init_time_segments(self, waypoints, total_time):
        differences = waypoints[:, 1:] - waypoints[:, :-1]
        distances = torch.sqrt(torch.sum(differences ** 2, dim=0))
        time_fraction = total_time / torch.sum(distances)
        arranged_time = torch.cat([torch.tensor([0]), torch.cumsum(distances * time_fraction, dim=0)])
        return arranged_time

    def evaluate_trajectory(self, T, plot_trajectory=False, iteration=None):
        polys_x = self.solve_minimum_snap(self.waypoints[0], T, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
        polys_y = self.solve_minimum_snap(self.waypoints[1], T, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
        if plot_trajectory:
            TrajectoryPlotter.plot_trajectory(self.waypoints, polys_x, polys_y, T, iteration=iteration, clear_plot=False, flag=False)
        cost = self.evaluate_cost_function(polys_x, polys_y, T)
        return cost

    def evaluate_cost_function(self, polys_x, polys_y, T):
        total_cost = 0
        num_segments = polys_x.shape[1]
        for segment in range(num_segments):
            segment_duration = T[segment + 1] - T[segment]
            snap_x = polys_x[5, segment]  # Changed from polys_x[5] to polys_x[4] for snap
            snap_y = polys_y[5, segment]  # Changed from polys_y[5] to polys_y[4] for snap
            segment_cost = (snap_x ** 2 + snap_y ** 2) * segment_duration
            total_cost += segment_cost
        return total_cost

    def optimize_time_segments(self, waypoints, total_time):
        T = self.init_time_segments(waypoints, total_time)
        T = torch.nn.Parameter(T)

        h = 1e-5
        max_iterations = 50
        lr = 0.1
        loss_history = []

        for iteration in range(max_iterations):
            loss = self.evaluate_trajectory(T, plot_trajectory=True, iteration=iteration + 1)

            with torch.no_grad():
                gradients = torch.zeros_like(T)
                for i in range(len(T) - 1):
                    gi = torch.ones_like(T) * -1 / (len(T) - 2)
                    gi[i] = 1
                    T_h = T + h * gi
                    gradients[i] = (self.evaluate_trajectory(T_h) - loss) / h

                step_size = lr
                for i in range(len(T) - 1):
                    while True:
                        new_T = T.clone()
                        new_T[i] -= step_size * gradients[i]
                        new_T.clamp_(min=0)
                        new_T /= new_T.sum()
                        new_T *= total_time
                        new_loss = self.evaluate_trajectory(new_T)
                        if new_loss < loss:
                            T.data = new_T
                            break
                        else:
                            step_size *= 0.5

                T.clamp_(min=0)
                T /= T.sum()
                T *= total_time

            print(f'Iteration {iteration + 1}, Loss: {loss.item()}, Updated Times: {T.data}')
            loss_history.append(loss.item())

        return T, loss_history

    def solve_minimum_snap(self, waypoints, time_stamps, poly_order, start_vel, start_acc, end_vel, end_acc):
        start_pos = waypoints[0]
        end_pos = waypoints[-1]
        num_segments = len(waypoints) - 1
        num_coefficients = poly_order + 1

        Q_all = torch.block_diag(*[TrajectoryUtils.compute_Q_matrix(poly_order, 3, time_stamps[i], time_stamps[i + 1]) for i in range(num_segments)])
        Q_all += torch.eye(Q_all.size(0), dtype=torch.float64) * 1e-6  # Ensure Q is SPD
        b_all = torch.zeros(Q_all.shape[0])

        Aeq = torch.zeros(4 * num_segments + 2, num_coefficients * num_segments)
        beq = torch.zeros(4 * num_segments + 2)
        Aeq[0:3, :num_coefficients] = torch.stack([
            TrajectoryUtils.calculate_time_vector(time_stamps[0], poly_order, 0),
            TrajectoryUtils.calculate_time_vector(time_stamps[0], poly_order, 1),
            TrajectoryUtils.calculate_time_vector(time_stamps[0], poly_order, 2)])
        Aeq[3:6, -num_coefficients:] = torch.stack([
            TrajectoryUtils.calculate_time_vector(time_stamps[-1], poly_order, 0),
            TrajectoryUtils.calculate_time_vector(time_stamps[-1], poly_order, 1),
            TrajectoryUtils.calculate_time_vector(time_stamps[-1], poly_order, 2)])
        beq[0:6] = torch.tensor([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc])

        num_eq_constraints = 6
        for i in range(1, num_segments):
            Aeq[num_eq_constraints, i * num_coefficients:(i + 1) * num_coefficients] = TrajectoryUtils.calculate_time_vector(time_stamps[i], poly_order, 0)
            beq[num_eq_constraints] = waypoints[i]
            num_eq_constraints += 1

        for i in range(1, num_segments):
            time_vector_p = TrajectoryUtils.calculate_time_vector(time_stamps[i], poly_order, 0)
            time_vector_v = TrajectoryUtils.calculate_time_vector(time_stamps[i], poly_order, 1)
            time_vector_a = TrajectoryUtils.calculate_time_vector(time_stamps[i], poly_order, 2)
            Aeq[num_eq_constraints:num_eq_constraints + 3, (i - 1) * num_coefficients:(i + 1) * num_coefficients] = torch.stack([
                torch.cat([time_vector_p, -time_vector_p]),
                torch.cat([time_vector_v, -time_vector_v]),
                torch.cat([time_vector_a, -time_vector_a])])
            num_eq_constraints += 3

        G_dummy = torch.zeros(1, Q_all.size(0), Q_all.size(0), dtype=torch.float64)
        h_dummy = torch.zeros(1, Q_all.size(0), dtype=torch.float64)
        Q_all = Q_all.to(dtype=self.dtype, device=self.device)
        b_all = b_all.to(dtype=self.dtype, device=self.device)
        Aeq = Aeq.to(dtype=self.dtype, device=self.device)
        beq = beq.to(dtype=self.dtype, device=self.device)
        G_dummy = G_dummy.to(dtype=self.dtype, device=self.device)
        h_dummy = h_dummy.to(dtype=self.dtype, device=self.device)

        solver_options = {'eps': 1e-24, 'maxIter': 100, 'solver': qpth.qp.QPSolvers.PDIPM_BATCHED}
        solution = QPFunction(verbose=-1, **solver_options)(Q_all, b_all, G_dummy, h_dummy, Aeq, beq)
        polynomial_coefficients = solution.view(num_segments, num_coefficients).transpose(0, 1)
        return polynomial_coefficients

    def forward(self, waypoints, total_time):
        optimized_time_segments, loss_history = self.optimize_time_segments(waypoints, total_time)
        polys_x = self.solve_minimum_snap(waypoints[0], optimized_time_segments, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
        polys_y = self.solve_minimum_snap(waypoints[1], optimized_time_segments, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
        return polys_x, polys_y, optimized_time_segments, loss_history

def demo_minimum_snap_simple():
    waypoints = torch.tensor([[0, 0], [1, 0], [1, 2], [0,2]], dtype=torch.float64).t()
    start_vel, start_acc, end_vel, end_acc = torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0])
    total_time = 25.0
    poly_order = 5  # Increased polynomial order

    planner = UAVTrajectoryPlanner(waypoints, total_time, poly_order, start_vel, start_acc, end_vel, end_acc, dtype=torch.float64, device='cpu')
    polys_x, polys_y, time_stamps, loss_history = planner(waypoints, total_time)
    TrajectoryPlotter.plot_trajectory(waypoints, polys_x, polys_y, time_stamps, clear_plot=True, flag=True)

    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.title("Total Cost Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    demo_minimum_snap_simple()
