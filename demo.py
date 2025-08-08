import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""
Extended Kalman Filter for 3D Lorenz Attractor

The Lorenz system's chaotic nature makes it sensitive to Q matrix tuning,
which can lead to filter divergence - the core problem the paper solves.
"""

# Lorenz system parameters (standard chaotic regime)
SIGMA, RHO, BETA = 10.0, 28.0, 8.0/3.0

def lorenz_dynamics(state, t):
    """Lorenz system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz"""
    x, y, z = state
    return [SIGMA*(y-x), x*(RHO-z)-y, x*y-BETA*z]

def lorenz_jacobian(state):
    """Jacobian matrix for EKF linearization"""
    x, y, z = state
    return np.array([[-SIGMA, SIGMA, 0],
                     [RHO-z, -1, -x], 
                     [y, x, -BETA]])

class ExtendedKalmanFilter:
    """EKF implementation for nonlinear systems"""
    
    def __init__(self, x0, P0, Q, R, dt):
        self.x = np.array(x0, dtype=float)  # State estimate
        self.P = np.array(P0, dtype=float)  # Error covariance  
        self.Q = np.array(Q, dtype=float)   # Process noise
        self.R = np.array(R, dtype=float)   # Measurement noise
        self.dt = dt
        
    def predict(self):
        """Prediction step: propagate state and covariance"""
        # Nonlinear state propagation (Euler integration)
        f_x = np.array(lorenz_dynamics(self.x, 0))
        self.x = self.x + self.dt * f_x
        
        # Linear covariance propagation  
        F = lorenz_jacobian(self.x)
        Phi = np.eye(3) + self.dt * F
        self.P = Phi @ self.P @ Phi.T + self.Q
        
    def update(self, measurement):
        """Update step: incorporate measurement"""
        # Measurement prediction (observe all states)
        h_x = self.x  
        H = np.eye(3)
        
        # Innovation and covariance
        y = measurement - h_x
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain and update
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

def run_simulation(scenario="good"):
    """Run EKF simulation with different tuning scenarios"""
    
    # Generate true Lorenz trajectory
    t = np.arange(0, 10, 0.01)
    true_state0 = [1.0, 1.0, 1.0]
    true_trajectory = odeint(lorenz_dynamics, true_state0, t)
    
    # Add measurement noise
    measurements = true_trajectory + np.random.normal(0, 0.5, true_trajectory.shape)
    
    # EKF setup with different tuning
    if scenario == "good":
        # Well-tuned EKF
        x0 = [1.2, 1.2, 1.2]  # Close initial guess
        P0 = np.eye(3) * 1.0   # Reasonable uncertainty
        Q = np.eye(3) * 0.1    # Balanced process noise
        print("Running well-tuned EKF...")
        
    else: 
        # Poor tuning
        x0 = [10.0, -10.0, 50.0]  # Even worse initial guess
        P0 = np.eye(3) * 0.001    # Extremely overconfident  
        Q = np.eye(3) * 1e-6      # Virtually no process noise
        
        print("Running poorly-tuned EKF (demonstrates divergence)...")
    
    R = np.eye(3) * 1.0  # Measurement noise variance
    
    # Run EKF
    ekf = ExtendedKalmanFilter(x0, P0, Q, R, 0.01)
    estimates = []
    
    for i, measurement in enumerate(measurements):
        ekf.predict()
        ekf.update(measurement) 
        estimates.append(ekf.x.copy())
        
        # Check for divergence
        if np.trace(ekf.P) > 1000:
            print(f"Filter diverged at step {i}!")
            break
    
    estimates = np.array(estimates)
    
    # Calculate RMSE
    min_len = min(len(estimates), len(true_trajectory))
    rmse = np.sqrt(np.mean(np.linalg.norm(
        true_trajectory[:min_len] - estimates[:min_len], axis=1)**2))
    
    print(f"RMSE: {rmse:.4f}")
    print(f"Final covariance trace: {np.trace(ekf.P):.4f}")
    
    return true_trajectory, estimates, t[:len(estimates)]

def plot_results():
    """Compare well-tuned vs poorly-tuned EKF"""
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Run both scenarios
    true_traj, good_est, t_good = run_simulation("good")
    
    np.random.seed(42)  # Same noise for fair comparison
    _, bad_est, t_bad = run_simulation("divergent")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 3D trajectories
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 'b-', label='True', alpha=0.7)
    ax.plot(good_est[:, 0], good_est[:, 1], good_est[:, 2], 'g--', label='Well-tuned EKF')
    ax.set_title('Well-tuned EKF')
    ax.legend()
    
    ax = fig.add_subplot(222, projection='3d')
    min_len = min(len(true_traj), len(bad_est))
    ax.plot(true_traj[:min_len, 0], true_traj[:min_len, 1], true_traj[:min_len, 2], 'b-', label='True', alpha=0.7)
    ax.plot(bad_est[:, 0], bad_est[:, 1], bad_est[:, 2], 'r--', label='Divergent EKF')
    ax.set_title('Poorly-tuned EKF (Divergent)')
    ax.legend()
    
    # Time series comparison
    axes[1,0].plot(t_good, true_traj[:len(good_est), 0], 'b-', label='True X')
    axes[1,0].plot(t_good, good_est[:, 0], 'g--', label='EKF X')
    axes[1,0].set_title('X State: Well-tuned')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    axes[1,1].plot(t_bad, true_traj[:len(bad_est), 0], 'b-', label='True X')
    axes[1,1].plot(t_bad, bad_est[:, 0], 'r--', label='EKF X')
    axes[1,1].set_title('X State: Divergent')
    axes[1,1].legend() 
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("EKF for Lorenz Attractor - Demonstrating Tuning Sensitivity")
    print("=" * 55)
    plot_results()
    
    print("\n" + "=" * 55)
    print("KEY INSIGHTS:")
    print("1. EKF performance is highly sensitive to Q matrix tuning")
    print("2. Poor tuning leads to filter divergence in chaotic systems") 
    print("3. Manual tuning is challenging and problem-specific")

# --- CONNECTION TO EKF-EM PARADIGM ---
#
# This demonstration shows exactly the problem that Saha & Ghosh's 
# EKF-EM paradigm addresses:
#
# THE PROBLEM: 
# - Standard EKF requires manual Q-R tuning
# - Poor tuning causes divergence (shown above)
# - Trial-and-error tuning is inefficient and unreliable
#
# THE EKF-EM SOLUTION:
# - Provides systematic method to tune Q adaptively  
# - Uses robustness & sensitivity metrics to guide tuning
# - Prevents divergence through intelligent Q adjustment
# - Eliminates manual tuning burden
#
# IMPLEMENTATION POTENTIAL:
# The EKF class above could be extended with EKF-EM by:
# 1. Adding innovation sequence analysis 
# 2. Computing robustness/sensitivity metrics at each step
# 3. Updating Q matrix based on these metrics
# 4. Maintaining balanced filter performance automatically
#
# This would transform the filter from static to adaptive,
# solving the divergence problem demonstrated here.