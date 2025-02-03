import numpy as np

def xyxy_to_cxcywh(xyxy):
    """
    Convert [x1, y1, x2, y2] -> [cx, cy, w, h].
    """
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1)
    h = (y2 - y1)
    return np.array([cx, cy, w, h], dtype=np.float32)

def cxcywh_to_xyxy(cxcywh):
    """
    Convert [cx, cy, w, h] -> [x1, y1, x2, y2].
    """
    cx, cy, w, h = cxcywh
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return [float(x1), float(y1), float(x2), float(y2)]

class KalmanBoxTracker2:
    """
    A simple Kalman Filter tracker for bounding boxes (xyxy externally),
    but internally uses center, width, height, and velocities.
    
    Internal state vector (8D):
        [cx, cy, w, h, vx, vy, vw, vh]^T
    where (cx, cy) is the box center, (w, h) are width and height,
    and (vx, vy, vw, vh) are their velocities.
    """

    def __init__(self, init_box_xyxy, init_time,
                 process_cov=1e-4, 
                 measure_cov=1e-2):
        """
        Initialize the tracker with the first bounding box measurement
        and the initial time in seconds.

        Args:
            init_box_xyxy (list or np.ndarray): [x1, y1, x2, y2] in normalized xyxy format
            init_time (float): time stamp for the initial detection
            process_cov (float): process noise scale (Q)
            measure_cov (float): measurement noise scale (R)
        """
        # Convert from xyxy to center/width/height
        cxcywh = xyxy_to_cxcywh(init_box_xyxy)

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[0, 0] = cxcywh[0]  # cx
        self.x[1, 0] = cxcywh[1]  # cy
        self.x[2, 0] = cxcywh[2]  # w
        self.x[3, 0] = cxcywh[3]  # h
        # velocities initialized to 0
        self.x[4, 0] = 0.0  # vx
        self.x[5, 0] = 0.0  # vy
        self.x[6, 0] = 0.0  # vw
        self.x[7, 0] = 0.0  # vh

        # State covariance
        # Start with some uncertainty in position and a large uncertainty in velocity
        self.P = np.eye(8, dtype=np.float32)
        self.P[4:, 4:] *= 1000.  # Large uncertainty in velocity initially

        # Process noise covariance
        self.Q = process_cov * np.eye(8, dtype=np.float32)

        # Measurement noise covariance
        # We'll measure [cx, cy, w, h] -> dimension 4
        self.R = measure_cov * np.eye(4, dtype=np.float32)

        # Measurement matrix H: we observe (cx, cy, w, h) directly
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy
        self.H[2, 2] = 1.0  # w
        self.H[3, 3] = 1.0  # h

        # Store last time to compute delta time
        self.last_time = init_time

    def _build_F(self, dt):
        """
        Build the state transition matrix F for variable dt.
        We assume:
            cx' = cx + vx * dt
            cy' = cy + vy * dt
            w'  = w  + vw * dt
            h'  = h  + vh * dt
        velocities remain unchanged.
        """
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = dt  # cx depends on vx
        F[1, 5] = dt  # cy depends on vy
        F[2, 6] = dt  # w depends on vw
        F[3, 7] = dt  # h depends on vh
        return F

    def predict(self, current_time):
        """
        Predict step of the Kalman Filter. Uses the time difference
        between the last update/predict and current_time.

        Args:
            current_time (float): The current time in seconds.

        Returns:
            predicted_box_xyxy (list): Predicted bounding box in xyxy format.
        """
        dt = current_time - self.last_time
        F = self._build_F(dt)

        # Predict state: x = F * x
        self.x = F @ self.x

        # Predict covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

        # Update time
        self.last_time = current_time

        # Convert internal (cx, cy, w, h) to xyxy
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        predicted_box_xyxy = cxcywh_to_xyxy([cx, cy, w, h])
        return predicted_box_xyxy

    def update(self, measurement_box_xyxy, current_time):
        """
        Update step of the Kalman Filter using a new measurement.

        Args:
            measurement_box_xyxy (list or np.ndarray): [x1, y1, x2, y2] in xyxy format
            current_time (float): The current time in seconds.

        Returns:
            updated_box_xyxy (list): Updated bounding box in xyxy format.
        """
        # 1) First predict to the current time
        self.predict(current_time)

        # 2) Construct measurement in (cx, cy, w, h)
        z_cxcywh = xyxy_to_cxcywh(measurement_box_xyxy).reshape(4, 1)

        # 3) Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R  # residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 4) Update with measurement
        y = z_cxcywh - (self.H @ self.x)  # measurement residual
        self.x = self.x + K @ y

        # 5) Update the covariance
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        # Convert internal (cx, cy, w, h) to xyxy
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        updated_box_xyxy = cxcywh_to_xyxy([cx, cy, w, h])
        return updated_box_xyxy

    def get_state(self):
        """
        Return the current state as a bounding box (xyxy).
        Useful if you just want the filter's current best estimate
        without an immediate predict/update call.
        """
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        return cxcywh_to_xyxy([cx, cy, w, h])
    
#
# Example Usage:
#
# tracker = KalmanBoxTracker(init_box_xyxy=[0.1, 0.2, 0.2, 0.3], init_time=0.0)
# # Suppose some time passes, e.g., t = 0.05
# pred_xyxy = tracker.predict(current_time=0.05)
# print("Predicted (xyxy):", pred_xyxy)
# # Suppose we get a measurement at t = 0.05
# updated_xyxy = tracker.update([0.12, 0.22, 0.22, 0.31], current_time=0.05)
# print("Updated (xyxy):", updated_xyxy)
#



import numpy as np
import scipy.linalg

class KalmanFilterXYAH:
    """
    Simplified Kalman filter for tracking bounding boxes in image space.
    
    The 8D state vector is: (x, y, a, h, vx, vy, va, vh), where:
      - (x, y) is the bounding box center,
      - a is the aspect ratio,
      - h is the bounding box height,
      - (vx, vy, va, vh) are their respective velocities.
    
    Measurement space is 4D: (x, y, a, h).
    
    Attributes:
        _std_weight_position (float): Standard deviation weight for position terms.
        _std_weight_velocity (float): Standard deviation weight for velocity terms.
        _update_mat (np.ndarray): The linear observation model (shape 4x8).
    """
    
    def __init__(self,
                 std_weight_position: float = 1.0 / 20,
                 std_weight_velocity: float = 1.0 / 160):
        """
        Initialize the filter with default noise parameters.
        
        Args:
            std_weight_position (float): Controls the magnitude of the position noise.
            std_weight_velocity (float): Controls the magnitude of the velocity noise.
        """
        # Store noise parameters
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        
        # The measurement update matrix (maps 8D state to 4D measurement)
        self._update_mat = np.eye(4, 8)  # shape: (4, 8)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a new track from an unassociated measurement.
        
        Args:
            measurement (ndarray): [x, y, a, h]
        
        Returns:
            mean (ndarray): 8D mean [x, y, a, h, vx, vy, va, vh]
            covariance (ndarray): 8x8 covariance matrix
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel], axis=0)

        # Standard deviations for [x, y, a, h, vx, vy, va, vh]
        std = [
            2 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[3],  # y
            1e-2,                                            # a
            2 * self._std_weight_position * measurement[3],  # h
            10 * self._std_weight_velocity * measurement[3], # vx
            10 * self._std_weight_velocity * measurement[3], # vy
            1e-5,                                            # va
            10 * self._std_weight_velocity * measurement[3], # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self,
                mean: np.ndarray,
                covariance: np.ndarray,
                dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform the prediction step of the Kalman filter.
        
        Args:
            mean (ndarray): 8D mean state vector at previous time step.
            covariance (ndarray): 8x8 covariance matrix at previous time step.
            dt (float): Variable time step.
        
        Returns:
            predicted_mean (ndarray): Updated 8D mean.
            predicted_covariance (ndarray): Updated 8x8 covariance matrix.
        """
        # 1) Construct motion matrix F for variable dt
        #    shape: (8, 8). The top-right block is dt * I.
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = dt
        
        # 2) Compute process noise Q
        #    We assume the noise scales with measurement[3] (the height).
        #    If mean[3] is negative or zero, we rely on absolute value just in case.
        h = abs(mean[3]) if mean[3] <= 0 else mean[3]
        
        std_pos = [
            self._std_weight_position * h,  # x
            self._std_weight_position * h,  # y
            1e-2,                           # a
            self._std_weight_position * h,  # h
        ]
        std_vel = [
            self._std_weight_velocity * h,  # vx
            self._std_weight_velocity * h,  # vy
            1e-5,                           # va
            self._std_weight_velocity * h,  # vh
        ]
        # Square them to form the diagonal
        Q = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        # 3) Predict the new mean/cov
        predicted_mean = F @ mean
        predicted_covariance = F @ covariance @ F.T + Q

        return predicted_mean, predicted_covariance

    def project(self,
                mean: np.ndarray,
                covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project the 8D state distribution into 4D measurement space.
        
        Args:
            mean (ndarray): 8D mean state.
            covariance (ndarray): 8x8 state covariance.
        
        Returns:
            projected_mean (ndarray): 4D measurement mean.
            projected_covariance (ndarray): 4x4 measurement covariance.
        """
        # For measurement space, the noise is typically smaller on aspect ratio
        # but we set some constant plus position-dependent for x, y, h
        h = abs(mean[3]) if mean[3] <= 0 else mean[3]

        # Innovation noise in measurement space
        std = [
            self._std_weight_position * h,  # x
            self._std_weight_position * h,  # y
            1e-1,                           # a
            self._std_weight_position * h,  # h
        ]
        R = np.diag(np.square(std))
        
        # Project mean and covariance down to 4D
        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T
        
        return projected_mean, projected_cov + R

    def update(self,
               mean: np.ndarray,
               covariance: np.ndarray,
               measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform the update (correction) step of the Kalman filter with a new measurement.
        
        Args:
            mean (ndarray): 8D predicted mean.
            covariance (ndarray): 8x8 predicted covariance.
            measurement (ndarray): 4D measurement [x, y, a, h].
        
        Returns:
            updated_mean (ndarray): Corrected 8D mean state.
            updated_covariance (ndarray): Corrected 8x8 covariance matrix.
        """
        # 1) Project the predicted mean/cov down to measurement space
        proj_mean, proj_cov = self.project(mean, covariance)

        # 2) Compute the Kalman gain
        #    K = cov * H^T * (H * cov * H^T + R)^(-1)
        #    We'll do a Cholesky-based solve for better numerical stability
        chol_factor, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             (covariance @ self._update_mat.T).T,
                                             check_finite=False).T
        
        # 3) Compute the innovation and update the state
        innovation = measurement - proj_mean
        updated_mean = mean + innovation @ kalman_gain.T
        
        # 4) Update the covariance
        updated_covariance = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        
        return updated_mean, updated_covariance

    def multi_predict(self,
                      means: np.ndarray,
                      covariances: np.ndarray,
                      dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized prediction step for multiple objects.
        
        Args:
            means (ndarray): N x 8 matrix, each row is a mean state.
            covariances (ndarray): N x 8 x 8 array of state covariances.
            dt (float): Variable time step.
        
        Returns:
            predicted_means (ndarray): N x 8 matrix of predicted means.
            predicted_covariances (ndarray): N x 8 x 8 array of predicted covariances.
        """
        N = len(means)
        if N == 0:
            return means, covariances
        
        # 1) Construct the motion matrix F for variable dt
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = dt
        
        # 2) For each object, compute process noise Q
        #    We'll store them in a list and then stack.
        Qs = []
        for i in range(N):
            h = abs(means[i, 3]) if means[i, 3] <= 0 else means[i, 3]
            std_pos = [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
            ]
            std_vel = [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ]
            Q = np.diag(np.square(np.concatenate([std_pos, std_vel])))
            Qs.append(Q)
        Qs = np.stack(Qs, axis=0)  # shape: N x 8 x 8

        # 3) Predict means/covariances
        predicted_means = means @ F.T
        predicted_covariances = F @ covariances @ F.T  # shape: N x 8 x 8, but must handle batch
        # To handle batch form: F is 8x8, covariances is Nx8x8, so we do:
        # F @ cov_i @ F.T for each i in the batch
        # One way is tensordot or repeated multiplication:
        #   for i in range(N):
        #       predicted_covariances[i] = F @ covariances[i] @ F.T
        # but we can do it in a vectorized manner:
        predicted_covariances = np.einsum('ij,njk,kl->nil', F, covariances, F)
        predicted_covariances += Qs

        return predicted_means, predicted_covariances
    
class KalmanBoxTracker:
    def xyxy_to_cxcyha(self, xyxy):
        w=xyxy[2]-xyxy[0]
        h=xyxy[3]-xyxy[1]
        cx=0.5*(xyxy[0]+xyxy[2])
        cy=0.5*(xyxy[1]+xyxy[3])
        return [cx,cy,w/h,h]

    def __init__(self, init_box_xyxy, init_time):
        self.kf = KalmanFilterXYAH()
        measurement=self.xyxy_to_cxcyha(init_box_xyxy)
        self.mean, self.covariance=self.kf.initiate(measurement)
        self.time=init_time
    
    def predict_meancov(self, current_time):
        dt=(current_time-self.time)*7.5
        mean, cov = self.kf.predict(self.mean, self.covariance, dt=dt)
        return mean, cov
    
    def predict(self, current_time):
        mean, _=self.predict_meancov(current_time)
        cx=mean[0]
        cy=mean[1]
        a=mean[2]
        h=mean[3]
        w=a*h
        return [cx-w*0.5, cy-h*0.5, cx+w*0.5, cy+h*0.5]

    def update(self, measurement_box_xyxy, current_time):
        measurement=self.xyxy_to_cxcyha(measurement_box_xyxy)
        predicted_mean, predicted_covariance=self.predict_meancov(current_time)
        self.mean, self.covariance = self.kf.update(predicted_mean,
                                                    predicted_covariance,
                                                    measurement)