import torch
import torch.optim as optim
from qpth.qp import QPFunction

class SparsePolynomial:
    def __init__(self):
        self.coefficients = {}  # Dictionary to hold non-zero coefficients

    def set_coefficient(self, power, coefficient):
        if coefficient != 0:
            self.coefficients[power] = coefficient

    def evaluate(self, time, derivative_order=0):
        value = 0.0
        for power, coef in self.coefficients.items():
            if power >= derivative_order:
                value += coef * torch.prod(torch.arange(power - derivative_order + 1, power + 1).float()) * time ** (power - derivative_order)
        return value

    def snap_loss(self):
        snap = 0.0
        for power, coef in self.coefficients.items():
            if power >= 4:
                snap += (coef ** 2) * torch.prod(torch.arange(power - 3, power + 1).float()) ** 2
        return snap

class SparseTrajectoryUtils:
    @staticmethod
    def evaluate_sparse_polynomial(sparse_poly, time, derivative_order=0):
        value = 0.0
        for power, coef in sparse_poly.coefficients.items():
            if power >= derivative_order:
                value += coef * torch.prod(torch.arange(power - derivative_order + 1, power + 1).float()) * time ** (power - derivative_order)
        return value

    @staticmethod
    def snap_loss(sparse_polys):
        total_snap = 0.0
        for segment in sparse_polys:
            for poly in segment:
                total_snap += poly.snap_loss()
        return total_snap

class UAVTrajectoryPlanner:
    def __init__(self, waypoints, total_time, poly_order, start_vel, start_acc, end_vel, end_acc, dtype=torch.float64, device='cpu'):
        self.waypoints = waypoints.to(dtype=dtype, device=device)
        self.total_time = total_time
        self.poly_order = poly_order
        self.start_vel = start_vel.to(dtype=dtype, device=device)
        self.start_acc = start_acc.to(dtype=dtype, device=device)
        self.end_vel = end_vel.to(dtype=dtype, device=device)
        self.end_acc = end_acc.to(dtype=dtype, device=device)
        self.device = device

    def solve_minimum_snap(self, waypoints, time_segments, poly_order, start_vel, start_acc, end_vel, end_acc):
        num_segments = len(waypoints) - 1
        sparse_polys = [[SparsePolynomial() for _ in range(3)] for _ in range(num_segments)]

        for segment in range(num_segments):
            for dim in range(3):
                H = 2 * torch.eye(poly_order + 1, device=self.device)
                f = torch.zeros(poly_order + 1, device=self.device)
                A = torch.zeros(4, poly_order + 1, device=self.device)  # Position and velocity constraints
                b = torch.zeros(4, device=self.device)
                
                t0 = time_segments[segment]
                t1 = time_segments[segment + 1]
                
                # Position constraints at t0 and t1
                A[0, :] = torch.tensor([t0 ** i for i in range(poly_order + 1)], device=self.device)
                A[1, :] = torch.tensor([t1 ** i for i in range(poly_order + 1)], device=self.device)
                b[0] = waypoints[dim, segment]
                b[1] = waypoints[dim, segment + 1]
                
                # Velocity constraints at t0 and t1
                A[2, 1:] = torch.tensor([i * t0 ** (i - 1) for i in range(1, poly_order + 1)], device=self.device)
                A[3, 1:] = torch.tensor([i * t1 ** (i - 1) for i in range(1, poly_order + 1)], device=self.device)
                b[2] = 0  # Assuming zero velocity at waypoints for simplicity
                b[3] = 0
                
                # Use QPFunction from qpth which expects exactly 6 arguments
                G = torch.Tensor().to(self.device)
                h = torch.Tensor().to(self.device)
                x = QPFunction()(H, f, A, b, G, h)
                
                for power in range(poly_order + 1):
                    coef = x[power]
                    sparse_polys[segment][dim].set_coefficient(power, coef.item())

        return sparse_polys

    def evaluate_trajectory(self, sparse_polys, times):
        trajectory = []
        for t in times:
            point = [SparseTrajectoryUtils.evaluate_sparse_polynomial(sparse_poly, t) for sparse_poly in sparse_polys]
            trajectory.append(point)
        return torch.tensor(trajectory, dtype=torch.float64)

    def optimize_trajectory(self, waypoints, total_time, weights):
        waypoints.requires_grad_(True)
        optimizer = optim.Adam([waypoints], lr=0.01)
        loss_history = []

        for _ in range(100):
            optimizer.zero_grad()
            time_segments = torch.linspace(0, total_time, len(waypoints[0])).to(waypoints.device)
            sparse_polys = self.solve_minimum_snap(waypoints, time_segments, self.poly_order, self.start_vel, self.start_acc, self.end_vel, self.end_acc)
            loss = self.total_loss(sparse_polys, weights)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        return sparse_polys, loss_history

    def total_loss(self, sparse_polys, weights):
        snap = SparseTrajectoryUtils.snap_loss(sparse_polys)
        return weights[0] * snap

class TrajectoryPlotter:
    @staticmethod
    def plot_trajectory(waypoints, sparse_polys, time_segments, clear_plot=True, flag=True):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if clear_plot:
            ax.cla()

        times = torch.linspace(0, time_segments[-1], 1000)
        x_vals = [SparseTrajectoryUtils.evaluate_sparse_polynomial(sparse_polys[0][0], t) for t in times]
        y_vals = [SparseTrajectoryUtils.evaluate_sparse_polynomial(sparse_polys[0][1], t) for t in times]
        z_vals = [SparseTrajectoryUtils.evaluate_sparse_polynomial(sparse_polys[0][2], t) for t in times]

        ax.plot(x_vals, y_vals, z_vals, label='Trajectory')
        ax.scatter(waypoints[0].cpu(), waypoints[1].cpu(), waypoints[2].cpu(), color='r', marker='o', label='Waypoints')
        ax.legend()
        plt.show()

def demo_minimum_snap_simple():
    waypoints = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]], dtype=torch.float64).t()
    start_vel = torch.tensor([0, 0, 0], dtype=torch.float64)
    start_acc = torch.tensor([0, 0, 0], dtype=torch.float64)
    end_vel = torch.tensor([0, 0, 0], dtype=torch.float64)
    end_acc = torch.tensor([0, 0, 0], dtype=torch.float64)
    total_time = 25.0
    poly_order = 5  # Polynomial order
    weights = [1.0]  # Weight for snap

    planner = UAVTrajectoryPlanner(waypoints, total_time, poly_order, start_vel, start_acc, end_vel, end_acc, dtype=torch.float64, device='cpu')
    sparse_polys, loss_history = planner.optimize_trajectory(waypoints, total_time, weights=weights)

    # For plotting, we'll assume we're only plotting for the first segment
    time_segments = torch.linspace(0, total_time, len(waypoints[0]))

    TrajectoryPlotter.plot_trajectory(waypoints, sparse_polys, time_segments)

    # Plot the loss history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.title("Total Cost Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    demo_minimum_snap_simple()
