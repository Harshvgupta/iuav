import os
import time
import torch
import argparse
import pypose as pp
import matplotlib.pyplot as plt
from torch.linalg import cross

from pypose.lietensor.basics import vec2skew
# from pypose.module.pid import PID
import torch
from torch import nn
import numpy as np

class PID(nn.Module):

    def __init__(self, kp, ki, kd):
        super().__init__()
        self.integrity_initialized = False
        self.integity = None
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def forward(self, error, error_dot, ff=None):
        r"""
        Args:
            error: error value :math:`\mathbf{e}(t)` as the difference between a
                desired setpoint and a measured
                value.
            error_dot: The rate of the change of error value.
            ff: feedforward system input appened to the output of the pid controller
        """
        if not self.integrity_initialized:
            self.integity = torch.zeros_like(error)
            self.integrity_initialized = True

        self.integity += error

        if ff == None:
            ff = torch.zeros_like(error)

        return self.kp * error + self.ki * self.integity + self.kd * error_dot + ff

    def reset(self):
        r"""
        This method is used to reset the internal error integrity.
        """
        if self.integrity_initialized:
            self.integity = None
            self.last_error = None
            self.integrity_initialized = False



def angular_vel_2_quaternion_dot(quaternion, w):
    device = quaternion.device
    p, q, r = w

    zero_t = torch.zeros(1, device=device)

    omega_1 = torch.cat((zero_t, -r, q, -p))
    omega_2 = torch.cat((r, zero_t, -p, -q))
    omega_3 = torch.cat((-q, p, zero_t, -r))
    omega_4 = torch.cat((p, q, r, zero_t))

    omega_matrix = torch.stack((omega_1, omega_2, omega_3, omega_4))

    return -0.5 * omega_matrix @ quaternion.T


def skew2vec(input):
    # Convert batched skew matrices to vectors.
    return torch.vstack([-input[1, 2], input[0, 2], -input[0, 1]])


class MultiCopter(pp.module.NLS):
    def __init__(self, mass, g, J, dt):
        super(MultiCopter, self).__init__()
        self.device = J.device
        self.m = mass
        self.J = J
        self.J_inverse = torch.inverse(self.J)
        self.g = g
        self.tau = dt
        self.e3 = torch.tensor([[0., 0., 1.]], device=self.device).reshape(3, 1)

    def state_transition(self, state, input, t=None):
        new_state = self.rk4(state, input, self.tau)
        self.pose_normalize(new_state)
        return new_state

    def rk4(self, state, input, t=None):
        k1 = self.xdot(state, input)
        k1_state = state + k1 * t / 2
        self.pose_normalize(k1_state)

        k2 = self.xdot(k1_state, input)
        k2_state = state + k2 * t / 2
        self.pose_normalize(k2_state)

        k3 = self.xdot(k2_state, input)
        k3_state = state + k3 * t
        self.pose_normalize(k3_state)

        k4 = self.xdot(k3_state, input)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * t

    def pose_normalize(self, state):
        state[3:7] = state[3:7] / torch.norm(state[3:7])

    def observation(self, state, input, t=None):
        return state

    def xdot(self, state, input):
        pose, vel, angular_speed = state[3:7], state[7:10], state[10:13]
        thrust, M = input[0], input[1:4]

        # convert the 1d row vector to 2d column vector
        M = torch.unsqueeze(M, 0)
        pose = torch.unsqueeze(pose, 0)
        pose_SO3 = pp.LieTensor(pose, ltype=pp.SO3_type)
        Rwb = pose_SO3.matrix()[0]

        acceleration = (Rwb @ (-thrust * self.e3) + self.m * self.g * self.e3) / self.m

        angular_speed = torch.unsqueeze(angular_speed, 1)
        w_dot = self.J_inverse \
            @ (M.T - cross(angular_speed, self.J @ angular_speed, dim=0))

        # transfer angular_speed from body frame to world frame
        return torch.concat([
                vel,
                torch.squeeze(angular_vel_2_quaternion_dot(pose, angular_speed)),
                torch.squeeze(acceleration),
                torch.squeeze(w_dot)
            ]
        )


class GeometricController(torch.nn.Module):
    def __init__(self, parameters, mass, J, g):
        self.device = J.device
        self.parameters = parameters
        self.g = g
        self.m = mass
        self.J = J
        self.e3 = torch.tensor([0., 0., 1.], device=self.device).reshape(3, 1)

    def compute_pose_error(self, pose, ref_pose):
        err_pose =  ref_pose.T @ pose - pose.T @ ref_pose
        return 0.5 * torch.squeeze(skew2vec(err_pose), dim=0)

    def forward(self, state, ref_state):
        device = state.device
        des_pos = torch.unsqueeze(ref_state[0:3], 1)
        des_vel = torch.unsqueeze(ref_state[3:6], 1)
        des_acc = torch.unsqueeze(ref_state[6:9], 1)
        des_acc_dot = torch.unsqueeze(ref_state[9:12], 1)
        des_acc_ddot = torch.unsqueeze(ref_state[12:15], 1)
        des_b1 = torch.unsqueeze(ref_state[15:18], 1)
        des_b1_dot = torch.unsqueeze(ref_state[18:21], 1)
        des_b1_ddot = torch.unsqueeze(ref_state[21:24], 1)

        # extract specific state from state tensor
        position = torch.unsqueeze(state[0:3], 1)
        pose = state[3:7]
        vel = torch.unsqueeze(state[7:10], 1)
        angular_vel = torch.unsqueeze(state[10:13], 1)
        pose_Rwb = pp.LieTensor(pose, ltype=pp.SO3_type).matrix()

        # extract parameters
        kp, kv, kori, kw = self.parameters
        position_pid = PID(kp, 0, kv)
        pose_pid = PID(kori, 0, kw)

        # position controller
        des_b3 = - position_pid.forward(position - des_pos, vel - des_vel) \
            - self.m * self.g * self.e3 \
            + self.m * des_acc

        b3 = pose_Rwb @ self.e3
        thrust_des = torch.squeeze(-des_b3.T @ b3)

        # attitude controller
        err_vel_dot = self.g * self.e3 - thrust_des / self.m * b3 - des_acc
        des_b3_dot = - kp * (vel - des_vel) - kv * err_vel_dot + self.m * des_acc_dot

        # calculate des_b3, des_b3_dot, des_b3_ddot
        b3_dot = pose_Rwb @ vec2skew(torch.squeeze(angular_vel)) @ self.e3
        thrust_dot = torch.squeeze(- des_b3_dot.T @ b3 - des_b3.T @ b3_dot)
        err_vel_ddot = (-thrust_dot * b3 - thrust_des * b3_dot) / self.m - des_acc_dot
        des_b3_ddot = -kp * err_vel_dot - kv * err_vel_ddot + self.m * des_acc_ddot

        des_b3 = -des_b3 / torch.norm(des_b3)
        des_b3_dot = -des_b3_dot / torch.norm(des_b3_dot)
        des_b3_ddot = -des_b3_ddot / torch.norm(des_b3_ddot)

        # calculate des_b2, des_b2_dot, des_b3_ddot
        des_b2 = cross(des_b3, des_b1, dim=0)
        des_b2_dot = cross(des_b3_dot, des_b1, dim=0) + cross(des_b3, des_b1_dot, dim=0)
        des_b2_ddot = cross(des_b3_ddot, des_b1, dim=0) \
            + 2*cross(des_b3_dot, des_b1_dot, dim=0) \
            + cross(des_b3, des_b1_ddot, dim=0)
        des_b2 = des_b2 / torch.norm(des_b2)
        des_b2_dot = des_b2 / torch.norm(des_b2_dot)
        des_b2_ddot = des_b2 / torch.norm(des_b2_ddot)

        # calculate des_b1, des_b1_dot, des_b1_ddot
        des_b1 = cross(des_b2, des_b3, dim=0)
        des_b1_dot = cross(des_b2_dot, des_b3, dim=0) + cross(des_b2, des_b3_dot, dim=0)
        des_b1_ddot = cross(des_b2_ddot, des_b3, dim=0) \
            + 2 * cross(des_b2_dot, des_b3_dot, dim=0) \
            + cross(des_b2, des_b3_ddot, dim=0)
        des_b2 = des_b2 / torch.norm(des_b2)
        des_b1_dot = des_b2 / torch.norm(des_b1_dot)
        des_b1_ddot = des_b2 / torch.norm(des_b1_ddot)

        des_pose_Rwb = torch.concat([des_b1, des_b2, des_b3], dim=1)
        des_pose_Rwb_dot = torch.concat([des_b1_dot, des_b2_dot, des_b3_dot], dim=1)
        des_pose_Rwb_ddot = torch.concat([des_b1_ddot, des_b2_ddot, des_b3_ddot], dim=1)

        des_augular_vel = skew2vec(des_pose_Rwb.T @ des_pose_Rwb_dot)
        wedge_des_augular_vel = vec2skew(des_augular_vel.T)[0]
        des_augular_acc = skew2vec(des_pose_Rwb.T @ des_pose_Rwb_ddot
                                   - wedge_des_augular_vel @ wedge_des_augular_vel)

        M = - pose_pid.forward(self.compute_pose_error(pose_Rwb, des_pose_Rwb),
                               angular_vel - pose_Rwb.T @ (des_pose_Rwb @ des_augular_vel)) \
          + cross(angular_vel, self.J @ angular_vel, dim=0)
        temp_M = torch.squeeze(vec2skew(angular_vel.T)) \
          @ (pose_Rwb.T @ des_pose_Rwb @ des_augular_vel \
          - pose_Rwb.T @ des_pose_Rwb @ des_augular_acc)
        M = (M - self.J @ temp_M).reshape(-1)

        zero_force_tensor = torch.tensor([0.], device=device)
        return torch.concat([torch.max(zero_force_tensor, thrust_des), M])


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
        return values
def get_ref_states(dt, N, coeff_x, coeff_y, coeff_z, time_stamps, device):
    """
    Generate reference states based on the piecewise polynomial trajectory
    """
    ref_state = torch.zeros(N, 24, device=device)
    
    all_times = []
    for i in range(coeff_x.shape[1]):
        times = torch.arange(time_stamps[i], time_stamps[i + 1], dt , device=device)
        all_times.append(times)
    
    all_times = torch.cat(all_times)
    # Ensure the total number of time steps matches N
    if all_times.shape[0] != N:
        raise ValueError(f"Total time steps {all_times.shape[0]} do not match N {N}")

    ref_state[:, 0] = evaluate_polynomials(coeff_x, time_stamps, all_times, 0)
    ref_state[:, 1] = evaluate_polynomials(coeff_y, time_stamps, all_times, 0)
    ref_state[:, 2] = evaluate_polynomials(coeff_z, time_stamps, all_times, 0)

    ref_state[:, 3] = evaluate_polynomials(coeff_x, time_stamps, all_times, 1)
    ref_state[:, 4] = evaluate_polynomials(coeff_y, time_stamps, all_times, 1)
    ref_state[:, 5] = evaluate_polynomials(coeff_z, time_stamps, all_times, 1)

    ref_state[:, 6] = evaluate_polynomials(coeff_x, time_stamps, all_times, 2)
    ref_state[:, 7] = evaluate_polynomials(coeff_y, time_stamps, all_times, 2)
    ref_state[:, 8] = evaluate_polynomials(coeff_z, time_stamps, all_times, 2)

    ref_state[:, 9] = evaluate_polynomials(coeff_x, time_stamps, all_times, 3)
    ref_state[:, 10] = evaluate_polynomials(coeff_y, time_stamps, all_times, 3)
    ref_state[:, 11] = evaluate_polynomials(coeff_z, time_stamps, all_times, 3)

    ref_state[:, 12] = evaluate_polynomials(coeff_x, time_stamps, all_times, 4)
    ref_state[:, 13] = evaluate_polynomials(coeff_y, time_stamps, all_times, 4)
    ref_state[:, 14] = evaluate_polynomials(coeff_z, time_stamps, all_times, 4)

    # b1 axis orientation
    ref_state[:, 15:18] = torch.tensor([[1., 0., 0.]], device=device)
    # b1 axis orientation dot
    ref_state[:, 18:21] = torch.tensor([[0., 0., 0.]], device=device)
    # b1 axis orientation ddot
    ref_state[:, 21:24] = torch.tensor([[0., 0., 0.]], device=device)

    return ref_state





def subPlot(ax, x, y, style, xlabel=None, ylabel=None, label=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y, style, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Geometric controller Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/pid/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    time_stamps=torch.tensor([ 0.000,8.000,16.000,24.000], dtype=torch.float64)
    total_duration = time_stamps[-1] - time_stamps[0]

    dt = 0.01
    N = int(torch.ceil(total_duration / dt).item())
    
    # States: x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz
    state = torch.zeros(N, 13, device=args.device)
    state[0][6] = 1
    
    coeff_x=torch.tensor([[ 0.0000e+00,  6.5232e-01,  1.3870e+00],
        [-2.0903e-10, -4.1467e-01, -1.9037e+00],
        [ 0.0000e+00,  1.0512e-01,  4.3171e-01],
        [ 6.2791e-03, -7.0071e-03, -3.5137e-02],
        [-7.1485e-04,  1.2242e-04,  1.2095e-03],
        [ 2.1762e-05,  7.1449e-07, -1.5060e-05]], dtype=torch.float64) 
    coeff_y=torch.tensor([[ 0.0000e+00, -1.9523e+00,  3.8582e+00],
        [ 2.6368e-10,  1.2411e+00, -3.7676e+00],
        [-1.3097e-10, -3.1462e-01,  6.6529e-01],
        [-4.4954e-03,  3.5269e-02, -4.5260e-02],
        [ 8.4389e-04, -1.6620e-03,  1.3688e-03],
        [-3.5246e-05,  2.7748e-05, -1.5459e-05]], dtype=torch.float64) 
    coeff_z=torch.tensor([[ 0.0000e+00,  1.6756e+00, -1.9732e+00],
        [-3.1572e-10, -1.0651e+00,  1.6522e+00],
        [ 2.7756e-11,  2.7002e-01, -2.4398e-01],
        [ 7.9700e-03, -2.6157e-02,  1.5438e-02],
        [-1.0484e-03,  1.1022e-03, -4.4900e-04],
        [ 3.7038e-05, -1.7026e-05,  4.9495e-06]], dtype=torch.float64)
    
    
    all_times=[]
    for i in range(coeff_x.shape[1]):
        times = torch.arange(time_stamps[i], time_stamps[i + 1], 0.01)
        all_times.append(times)
    
    times1 = torch.cat(all_times)

    # Ensure the total number of time steps matches N
    if times1.shape[0] != N:
        raise ValueError(f"Total time steps {all_times.shape[0]} do not match N {N}")

    ref_state = get_ref_states(dt, N, coeff_x, coeff_y, coeff_z,time_stamps, args.device)

    # parameters = torch.ones(4, device=args.device) # kp, kv, kori, kw
    parameters = torch.tensor([0.5, 0.5,1.5, 0.5], device=args.device)
    mass = torch.tensor(0.18, device=args.device)
    g = torch.tensor(9.81, device=args.device)
    inertia = torch.tensor([[0.0820, 0., 0.00000255],
                            [0., 0.0845, 0.],
                            [0.00000255, 0., 0.1377]], device=args.device)

    controller = GeometricController(parameters, mass, inertia, g)
    model = MultiCopter(mass, g, inertia, dt).to(args.device)

    # Calculate trajectory
    for i in range(N - 1):
        state[i + 1], _ = model(state[i], controller.forward(state[i], ref_state[i]))
    tracking_error = torch.norm(state[:, 0:3] - ref_state[:, 0:3], dim=1).mean()
    print( 'Tracking Error:', {tracking_error})
    '''
    kp_range = torch.linspace(1.0, 2.0, 10, device=args.device)
    kv_range = torch.linspace(1.0, 2.0, 10, device=args.device)
    kori_range = torch.linspace(1.0, 2.0, 10, device=args.device)
    kw_range = torch.linspace(1.0, 2.0, 10, device=args.device)

    # Iterate over different combinations of gain values
    for kp in kp_range:
        for kv in kv_range:
            for kori in kori_range:
                for kw in kw_range:
                    parameters = torch.tensor([kp, kv, kori, kw], device=args.device)
                    controller = GeometricController(parameters, mass, inertia, g)
                    model = MultiCopter(mass, g, inertia, dt).to(args.device)

                    # Reset the initial state for each iteration
                    state = torch.zeros(N, 13, device=args.device)
                    state[0][6] = 1

                    # Calculate trajectory
                    for i in range(N - 1):
                        state[i + 1], _ = model(state[i], controller.forward(state[i], ref_state[i]))

                    # Evaluate the tracking performance
                    tracking_error = torch.norm(state[:, 0:3] - ref_state[:, 0:3], dim=1).mean()
                    print(f"Gains: kp={kp}, kv={kv}, kori={kori}, kw={kw}, Tracking Error: {tracking_error}")
    '''
    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=1, sharex=True)
    # subPlot(ax[0], times1, state[:, 0], '-', ylabel='X position (m)', label='true')
    # subPlot(ax[0], times1, ref_state[:, 0], '--', ylabel='X position (m)', label='sp')
    # subPlot(ax[1], times1, state[:, 1], '-', ylabel='Y position (m)', label='true')
    # subPlot(ax[1], times1, ref_state[:, 1], '--', ylabel='Y position (m)', label='sp')
    # subPlot(ax[2], times1, state[:, 2], '-', ylabel='Z position (m)', label='true')
    # subPlot(ax[2], times1, ref_state[:, 2], '--', ylabel='Z position (m)', label='sp')
    subPlot(ax[0], state[:,0], state[:, 1], '-', ylabel='X-Y position (m)', label='true')
    subPlot(ax[0], ref_state[:, 0], ref_state[:, 1], '--', ylabel='X-Y position (m)', label='sp')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    figure = os.path.join(args.save + 'geometric_controller.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
