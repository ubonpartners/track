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

    def predict_meancov(self, prediction_time):
        dt=(prediction_time-self.time)*7.5
        mean, cov = self.kf.predict(self.mean, self.covariance, dt=dt)
        return mean, cov

    def predict(self, prediction_time):
        mean, _=self.predict_meancov(prediction_time)
        cx=mean[0]
        cy=mean[1]
        a=mean[2]
        h=mean[3]
        w=a*h
        return [cx-w*0.5, cy-h*0.5, cx+w*0.5, cy+h*0.5]

    def update(self, measurement_box_xyxy, current_time, pf=None):
        measurement=self.xyxy_to_cxcyha(measurement_box_xyxy)
        if pf is not None:
            pf(f"KF update {[f'{x:.4f}' for x in measurement]}  T={current_time} dt={current_time-self.time}")

        predicted_mean, predicted_covariance=self.predict_meancov(current_time)
        self.mean, self.covariance = self.kf.update(predicted_mean,
                                                    predicted_covariance,
                                                    measurement)
        self.time=current_time