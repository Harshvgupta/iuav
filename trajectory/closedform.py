import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the dtype at one place
dtype = torch.float64

def demo3_minimum_snap_close_form():
    # Conditions
    waypts = torch.tensor([
        [0.0, 1.0, 2.0, 4.0, 5.0],
        [0.0, 2.0, -1.0, 8.0, 2.0]
    ], dtype=dtype)
    # waypts = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]], dtype=torch.float64).t()
    v0 = torch.tensor([0.0, 0.0], dtype=dtype)
    a0 = torch.tensor([0.0, 0.0], dtype=dtype)
    v1 = torch.tensor([0.0, 0.0], dtype=dtype)
    a1 = torch.tensor([0.0, 0.0], dtype=dtype)
    T = 5.0
    ts = arrange_T(waypts, T)
    print("ts:", ts)
    n_order = 5

    # Trajectory plan
    polys_x = minimum_snap_single_axis_close_form(waypts[0], ts, n_order, v0[0], a0[0], v1[0], a1[0])
    print("polys_x:", polys_x)
    polys_y = minimum_snap_single_axis_close_form(waypts[1], ts, n_order, v0[1], a0[1], v1[1], a1[1])
    print("polys_y:", polys_y)
    polys_z = minimum_snap_single_axis_close_form(waypts[2], ts, n_order, v0[1], a0[1], v1[1], a1[1])
    plot_trajectory(waypts, polys_x, polys_y,polys_z, ts)
    # # Result show
    # plt.figure(figsize=(10, 8))
    # plt.plot(waypts[0], waypts[1], '*r')
    # plt.plot(waypts[0], waypts[1], 'b--')
    # plt.title('Minimum Snap Trajectory')

    # colors = ['g', 'r', 'c']
    # for i in range(polys_x.shape[1]):
    #     tt = torch.arange(ts[i], ts[i+1], 0.01, dtype=dtype)
    #     xx = polys_vals(polys_x, ts, tt, 0)
    #     yy = polys_vals(polys_y, ts, tt, 0)
    #     plt.plot(xx, yy, color=colors[i % 3])

    # plt.show()

def evaluate_polynomial(polynomial_coefficients, time, derivative_order):
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

def evaluate_polynomials(polynomial_coefficients, time_stamps, times, derivative_order):
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
    print(f"time shapes are here {times.shape}")
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
            values[i] = evaluate_polynomial(polynomial_coefficients[:, index], time, derivative_order)
    print(f"the shape of the value is {values.shape}")
    return values

def plot_trajectory1(waypoints, polys_x, polys_y ,time_stamps):

    """
    Plot the minimum snap trajectory.
    
    Args:
        waypoints (Tensor): waypoints
        polys_x (Tensor): polynomial coefficients for x axis
        polys_y (Tensor): polynomial coefficients for y axis
        time_stamps (Tensor): time stamps for each segment
    """
    plt.plot(waypoints[0], waypoints[1], '*r')
    plt.plot(waypoints[0], waypoints[1], 'b--')
    plt.title('Minimum Snap Trajectory')
    colors = ['g', 'r', 'c']
    for i in range(polys_x.shape[1]):
        times = torch.arange(time_stamps[i], time_stamps[i+1], 0.01)
        x_values = evaluate_polynomials(polys_x, time_stamps, times, 0)
        y_values = evaluate_polynomials(polys_y, time_stamps, times, 0)
        # z_values = self.evaluate_polynomials(polys_z, time_stamps, times, 0)
        plt.plot(x_values.detach().numpy(), y_values.detach().numpy(),colors[i % 3])
    plt.show()

def plot_trajectory(waypoints, polys_x, polys_y, polys_z, time_stamps):
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
            x_values = evaluate_polynomials(polys_x, time_stamps, times, 0)
            y_values = evaluate_polynomials(polys_y, time_stamps, times, 0)
            z_values = evaluate_polynomials(polys_z, time_stamps, times, 0)
            ax.plot(x_values.detach().numpy(), y_values.detach().numpy(), z_values.detach().numpy(), color=colors[i % 3])
        print(f"the shape of the x_value is {x_values.shape}")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.title('Minimum Snap Trajectory')
        plt.show()
def arrange_T1(waypts, T):
    distances = torch.sqrt(torch.sum(torch.diff(waypts, dim=1) ** 2, dim=0))
    total_distance = torch.sum(distances)
    ts = torch.zeros(waypts.shape[1])
    ts[1:] = torch.cumsum(distances / total_distance * T, dim=0)
    return ts
def arrange_T( waypoints, total_time):
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

def polys_vals(polys, ts, tt, k):
    n_poly = polys.shape[1]
    vals = torch.zeros(tt.shape)
    for i in range(n_poly):
        mask = (tt >= ts[i]) & (tt <= ts[i + 1])
        t = tt[mask] - ts[i]
        n_coef = polys.shape[0]
        for j in range(k, n_coef):
            vals[mask] += polys[j, i] * torch.prod(torch.tensor(range(j - k + 1, j + 1), dtype=torch.float32)) * t ** (j - k)
    return vals

def minimum_snap_single_axis_close_form(wayp, ts, n_order, v0, a0, v1, a1):
    n_coef = n_order + 1
    n_poly = len(wayp) - 1
    print(n_poly)

    # Compute Q
    Q_all = torch.block_diag(*[compute_Q_matrix(n_order, 3, ts[i], ts[i + 1]) for i in range(n_poly)])
    # Q_all = torch.zeros((n_coef * n_poly, n_coef * n_poly))
    # for i in range(n_poly):
    #     Q_all[i*n_coef:(i+1)*n_coef, i*n_coef:(i+1)*n_coef] = compute_q(n_order, 3, ts[i], ts[i+1])
    # Q_all = torch.zeros((n_coef * n_poly, n_coef * n_poly), dtype=dtype)
    # for i in range(n_poly):
    #     Q = compute_Q(n_order, 3, ts[i], ts[i+1])
    #     Q_all[i*n_coef:(i+1)*n_coef, i*n_coef:(i+1)*n_coef] = Q

    # Scale Q_all by 1e6 to match MATLAB output
    # Q_all *= 1e6
    # print(Q_all)
    # Compute Tk
    tk = torch.zeros(n_poly + 1, n_coef, dtype=dtype)
    for i in range(n_coef):
        tk[:, i] = ts.pow(i)

    # Compute A
    n_continuous = 3
    A = torch.zeros(n_continuous * 2 * n_poly, n_coef * n_poly, dtype=dtype)
    for i in range(n_poly):
        for j in range(n_continuous):
            for k in range(j, n_coef):
                if k == j:
                    t1 = t2 = 1
                else:
                    t1 = tk[i, k-j]
                    t2 = tk[i+1, k-j]
                A[n_continuous*2*i+j, n_coef*i+k] = torch.prod(torch.arange(k-j+1, k+1, dtype=dtype)) * t1
                A[n_continuous*2*i+n_continuous+j, n_coef*i+k] = torch.prod(torch.arange(k-j+1, k+1, dtype=dtype)) * t2
    # print(A)

    # Compute M
    M = compute_M(n_poly, n_continuous)
    # print(M)

    # Compute C
    num_d = n_continuous * (n_poly + 1)
    C = torch.eye(num_d, dtype=dtype)
    df = torch.cat([wayp, torch.tensor([v0, a0, v1, a1], dtype=dtype)])
    fix_idx = torch.cat([torch.arange(0, num_d, 3, dtype=torch.int64), torch.tensor([1, 2, num_d-2, num_d-1], dtype=torch.int64)])
    free_idx = torch.tensor([i for i in range(num_d) if i not in fix_idx], dtype=torch.int64)
    C = torch.cat([C[:, fix_idx], C[:, free_idx]], dim=1)
    # print(C)
    AiMC = torch.linalg.inv(A) @ M @ C
    R = AiMC.T @ Q_all @ AiMC
    # print(R)
    n_fix = len(fix_idx)
    Rff = R[:n_fix, :n_fix]
    Rpp = R[n_fix:, n_fix:]
    Rfp = R[:n_fix, n_fix:]
    
    dp = -torch.linalg.inv(Rpp) @ Rfp.T @ df
    # print(dp)
    p = AiMC @ torch.cat([df, dp])
    # print(p)
    # p = p.view(n_coef, n_poly).transpose(0, 1)
    p = p.reshape(-1, 6).T
    # return p.reshape(n_coef, n_poly)
    return p

def compute_Q(n, r, t1, t2):
    T = torch.zeros((n-r)*2 + 1, dtype=dtype)
    for i in range((n-r)*2 + 1):
        T[i] = t2**(i+1) - t1**(i+1)

    Q = torch.zeros((n+1, n+1), dtype=dtype)
    for i in range(r, n+1):
        for j in range(i, n+1):
            k1 = i - r
            k2 = j - r
            k = k1 + k2 + 1
            Q[i, j] = torch.prod(torch.arange(k1+1, k1+r+1, dtype=dtype)) * \
                      torch.prod(torch.arange(k2+1, k2+r+1, dtype=dtype)) / k * T[k-1]
            Q[j, i] = Q[i, j]

    return Q

def compute_q(n_order, k, t1, t2):
    q = torch.zeros((n_order + 1, n_order + 1), dtype=dtype)
    for i in range(k, n_order + 1):
        for j in range(k, n_order + 1):
            q[i, j] = (
                torch.prod(torch.tensor(range(i - k + 1, i + 1), dtype=dtype))
                * torch.prod(torch.tensor(range(j - k + 1, j + 1), dtype=dtype))
                / (i + j - 2 * k + 1)
                * (t2 ** (i + j - 2 * k + 1) - t1 ** (i + j - 2 * k + 1))
            )
    return q

def compute_Q_matrix(poly_order, derivative_order, start_time, end_time):
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
        time_diff_powers = torch.zeros((poly_order - derivative_order) * 2 + 1, dtype=dtype)
        for i in range((poly_order - derivative_order) * 2 + 1):
            time_diff_powers[i] = end_time ** (i + 1) - start_time ** (i + 1)

        Q_matrix = torch.zeros(poly_order + 1, poly_order + 1, dtype=dtype)
        for i in range(derivative_order + 1, poly_order + 2):
            for j in range(i, poly_order + 2):
                k1 = i - derivative_order - 1
                k2 = j - derivative_order - 1
                k = k1 + k2 + 1
                prod_k1 = torch.prod(torch.tensor(range(k1 + 1, k1 + derivative_order + 1), dtype=dtype))
                prod_k2 = torch.prod(torch.tensor(range(k2 + 1, k2 + derivative_order + 1), dtype=dtype)) 
                Q_matrix[i - 1, j - 1] = prod_k1 * prod_k2 / k * time_diff_powers[k - 1]
                Q_matrix[j - 1, i - 1] = Q_matrix[i - 1, j - 1]

        return Q_matrix

def compute_M(n_poly, n_continuous):
    num_d = n_continuous * (n_poly + 1)
    M = torch.zeros(n_poly * 2 * n_continuous, num_d, dtype=dtype)
    
    for i in range(n_poly):
        start_row = i * 2 * n_continuous
        start_col = i * n_continuous
        
        # First set of rows for each polynomial
        M[start_row:start_row + n_continuous, start_col:start_col + n_continuous] = torch.eye(n_continuous, dtype=dtype)
        
        # Second set of rows for each polynomial
        M[start_row + n_continuous:start_row + 2*n_continuous, start_col + n_continuous:start_col + 2*n_continuous] = torch.eye(n_continuous, dtype=dtype)
    
    return M

# def polys_vals(polys, ts, tt, n):
#     # Implement this function if needed
#     # For now, we'll return a placeholder
#     return torch.ones_like(tt, dtype=dtype)

# Run the demo
demo3_minimum_snap_close_form()
