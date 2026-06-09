import math
import numpy as np  


#############################################################
# MAIN RELU MRAC CLASS DEFINITION TO BE USED IN relu_mrac.py
#############################################################
class M_ReLuMRAC:
  @staticmethod
  def computeControlLaw(
    K_hat_x: np.ndarray,
    x: np.ndarray,
    K_hat_r: np.ndarray,
    r: np.ndarray,
    Theta_hat: np.ndarray,
    Phi: np.ndarray
    ) -> np.ndarray:
    """
    Compute the classical MRAC control law.
    """
    control_input = (
      K_hat_x.T * x
      + K_hat_r.T * r
      - Theta_hat.T * Phi
    )

    return control_input
  
  @staticmethod
  def computeControlLawReLu(
    K_hat_x: np.ndarray,
    x: np.ndarray,
    K_hat_r: np.ndarray,
    r: np.ndarray,
    Theta_hat: np.ndarray,
    Phi: np.ndarray,
    Theta_hat_relu: np.ndarray,
    Phi_relu: np.ndarray
    ) -> np.ndarray:
    """
    Compute the classical MRAC control law.
    """
    control_input = (
      K_hat_x.T * x
      + K_hat_r.T * r
      - Theta_hat.T * Phi
      - Theta_hat_relu.T * Phi_relu
    )

    return control_input

  @staticmethod
  def computeAdaptiveLaw(
      Gamma_gain: np.ndarray,
      pi_vector: np.ndarray,
      eTranspose_P_B: np.ndarray
    ) -> np.ndarray:
    """
    Compute the classical MRAC adaptive law.
    """
    K_hat_state_dot = Gamma_gain * pi_vector * eTranspose_P_B
    return K_hat_state_dot

  @staticmethod
  def compute_eTransposePB(e: np.ndarray, P: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute eᵀ * P * B and its norm.

    Parameters:
      e (np.ndarray): Error vector (column vector).
      P (np.ndarray): Symmetric positive definite matrix.
      B (np.ndarray): Input matrix.

    Returns:
      tuple:
        - eTranspose_P_B (np.ndarray): The product eᵀ * P * B (1 x m vector if B has m columns).
        - eTranspose_P_B_norm (float): The Euclidean norm (2-norm) of eᵀ * P * B.
    """
    eTranspose_P_B = e.T * P * B
    eTranspose_P_B_norm = float(np.linalg.norm(eTranspose_P_B))
    return eTranspose_P_B, eTranspose_P_B_norm

  @staticmethod
  def computeAllAdaptiveLaws(
    Gamma_x,
    x,
    Gamma_r,
    r,
    Gamma_Theta,
    Phi_regressor_vector,
    eTranspose_P_B
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all the classical MRAC adaptive laws

    Returns:
      - (K_hat_x_dot, K_hat_r_dot, Theta_hat_dot)
    """

    K_hat_x_dot = M_ReLuMRAC.computeAdaptiveLaw(-Gamma_x, x, eTranspose_P_B)
    K_hat_r_dot = M_ReLuMRAC.computeAdaptiveLaw(-Gamma_r, r, eTranspose_P_B)
    Theta_hat_dot = M_ReLuMRAC.computeAdaptiveLaw(Gamma_Theta, Phi_regressor_vector, eTranspose_P_B)

    return (K_hat_x_dot, K_hat_r_dot, Theta_hat_dot)
  
  @staticmethod
  def deadZoneModulationFunction(e_vector: np.ndarray, delta: float, e_0: float, use_deadzone: bool) -> float:
    """
    Smooth dead-zone modulation function for MRAC.

    Reference: E. Lavretsky, K. Wise, "Robust and Adaptive Control", Springer 2013, Sec. 11.2.1

    Parameters:
      - e_vector (np.ndarray): Tracking error vector.
      - delta (float): Must satisfy 0 < delta < 1. Characterizes the slope of the modulation function.
      - e_0 (float): Must be > 0. Defines the dead-zone threshold. The dead-zone modification stops the
      adaptation process when the norm of the tracking error becomes smaller than the prescribed value e_0.
      - use_deadzone (bool): If False, the function returns 1.0 (i.e., no dead-zone effect).

    Returns:
      float: Modulation coefficient between 0.0 and 1.0.
    """
    if not use_deadzone:
      return 1.0

    norm_e = np.linalg.norm(e_vector)
    coeff = (norm_e - delta * e_0) / ((1.0 - delta) * e_0)
    result = float(max(0.0, min(1.0, coeff)))
    return result
  
  @staticmethod
  def computeRobustAdaptiveLaw(
      Gamma_gain: np.ndarray,
      dead_zone_value: float,
      pi_vector: np.ndarray,
      eTranspose_P_B: np.ndarray,
      sigma_gain: float,
      eTranspose_P_B_norm: float,
      K_hat_state: np.ndarray,
      use_deadzone: bool,
      use_emodification: bool
    ) -> np.ndarray:
    """
    Compute the MRAC adaptive law with OPTIONAL dead-zone modification and e-modification capabilities.
    """
    modulation_factor = dead_zone_value if use_deadzone else 1.0
    classic_term = pi_vector * eTranspose_P_B

    if use_emodification:
      emod_term = sigma_gain * eTranspose_P_B_norm * K_hat_state
      update_term = classic_term - emod_term
    else:
      update_term = classic_term

    K_hat_state_dot = Gamma_gain * modulation_factor * update_term
    return K_hat_state_dot
  
  @staticmethod
  def computeAllRobustAdaptiveLawsReLu(
    Gamma_x: np.ndarray,
    x: np.ndarray,
    Gamma_r: np.ndarray,
    r: np.ndarray,
    Gamma_Theta: np.ndarray,
    Phi_regressor_vector: np.ndarray,
    Gamma_Theta_ReLu: np.ndarray,
    Phi_regressor_NN: np.ndarray,
    eTranspose_P_B: np.ndarray,
    dead_zone_value: float,
    sigma_x: float,
    sigma_r: float,
    sigma_Theta: float,
    sigma_Theta_ReLu: float,
    eTranspose_P_B_norm: float,
    K_hat_x: np.ndarray,
    K_hat_r: np.ndarray,
    Theta_hat: np.ndarray,
    Theta_hat_ReLu: np.ndarray,
    use_deadzone: bool,
    use_emodification: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all MRAC adaptive laws with OPTIONAL dead-zone and e-modification.

    Returns:
      - (K_hat_x_dot, K_hat_r_dot, Theta_hat_dot)
    """
    K_hat_x_dot = M_ReLuMRAC.computeRobustAdaptiveLaw(
      -Gamma_x,
      dead_zone_value,
      x,
      eTranspose_P_B,
      sigma_x,
      eTranspose_P_B_norm,
      K_hat_x,
      use_deadzone,
      use_emodification
    )

    K_hat_r_dot = M_ReLuMRAC.computeRobustAdaptiveLaw(
      -Gamma_r,
      dead_zone_value,
      r,
      eTranspose_P_B,
      sigma_r,
      eTranspose_P_B_norm,
      K_hat_r,
      use_deadzone,
      use_emodification
    )

    Theta_hat_dot = M_ReLuMRAC.computeRobustAdaptiveLaw(
      Gamma_Theta,
      dead_zone_value,
      Phi_regressor_vector,
      eTranspose_P_B,
      sigma_Theta,
      eTranspose_P_B_norm,
      Theta_hat,
      use_deadzone,
      use_emodification
    )

    Theta_hat_ReLu_dot = M_ReLuMRAC.computeRobustAdaptiveLaw(
       Gamma_Theta_ReLu,
       dead_zone_value,
       Phi_regressor_NN,
       eTranspose_P_B,
       sigma_Theta_ReLu,
       eTranspose_P_B_norm,
       Theta_hat_ReLu,
       use_deadzone,
       use_emodification
    )

    return (K_hat_x_dot, K_hat_r_dot, Theta_hat_dot, Theta_hat_ReLu_dot)
  
  @staticmethod
  def computeAllRobustAdaptiveLaws(
    Gamma_x: np.ndarray,
    x: np.ndarray,
    Gamma_r: np.ndarray,
    r: np.ndarray,
    Gamma_Theta: np.ndarray,
    Phi_regressor_vector: np.ndarray,
    eTranspose_P_B: np.ndarray,
    dead_zone_value: float,
    sigma_x: float,
    sigma_r: float,
    sigma_Theta: float,
    eTranspose_P_B_norm: float,
    K_hat_x: np.ndarray,
    K_hat_r: np.ndarray,
    Theta_hat: np.ndarray,
    use_deadzone: bool,
    use_emodification: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all MRAC adaptive laws with OPTIONAL dead-zone and e-modification.

    Returns:
      - (K_hat_x_dot, K_hat_r_dot, Theta_hat_dot)
    """
    K_hat_x_dot = M_ReLuMRAC.computeRobustAdaptiveLaw(
      -Gamma_x,
      dead_zone_value,
      x,
      eTranspose_P_B,
      sigma_x,
      eTranspose_P_B_norm,
      K_hat_x,
      use_deadzone,
      use_emodification
    )

    K_hat_r_dot = M_ReLuMRAC.computeRobustAdaptiveLaw(
      -Gamma_r,
      dead_zone_value,
      r,
      eTranspose_P_B,
      sigma_r,
      eTranspose_P_B_norm,
      K_hat_r,
      use_deadzone,
      use_emodification
    )

    Theta_hat_dot = M_ReLuMRAC.computeRobustAdaptiveLaw(
      Gamma_Theta,
      dead_zone_value,
      Phi_regressor_vector,
      eTranspose_P_B,
      sigma_Theta,
      eTranspose_P_B_norm,
      Theta_hat,
      use_deadzone,
      use_emodification
    )

    return (K_hat_x_dot, K_hat_r_dot, Theta_hat_dot)
  
#############################################################
# RELU CLASS DEFINITION TO BE USED FOR REGRESSOR COMPUTATION
#############################################################
class ReLUFeatureMap:
    """
    Explicit ReLU deep neural network (DNN) feature map Φ(x).

    This class constructs and evaluates a fully-connected feedforward network
    with ReLU activations in all hidden layers and a linear readout layer.

    The network structure is:

        Hidden layer 1:
            z₁ = W₁ x + b₁
            a₁ = ReLU(z₁)

        Hidden layer l (for l = 2, ..., L):
            z_l = W_l a_{l-1} + b_l
            a_l = ReLU(z_l)

        Feature map:
            Φ(x) = a_L

    Optionally, a final linear readout is provided:

        y(x) = Θ Φ(x) + c

    where:
        - x ∈ Rⁿ is the input
        - Φ(x) ∈ R^{width} is the feature vector
        - y(x) ∈ Rᵐ is the output of the linear readout
        - W_l, Θ, b_l, c are randomly initialized parameters
    """

    def __init__(self, n, m, width, depth, seed=None):
        """
        Initialize the ReLU feature map network.

        Parameters
        ----------
        n : int
            Input dimension (size of x).
            The network expects inputs of shape (N, n) or (n,) where n is
            the number of state/feature components.
        m : int
            Output dimension of the optional linear readout y(x).
            This does NOT affect the dimension of Φ(x); it only defines
            the size of the final linear layer mapping Φ(x) → y(x).
        width : int
            Width of each hidden layer (number of neurons per hidden layer).
            This also equals the dimension of the feature vector Φ(x).
        depth : int
            Number of hidden layers in the network.
            depth ≥ 1: there is always at least one hidden layer.
        seed : int or None, optional
            Random seed for reproducible initialization.
            If None, NumPy will use its default RNG behavior.

        Notes
        -----
        The internal parameterization is:

            self.W[0] : (width, n)        # weights of first hidden layer
            self.b[0] : (width,)          # biases of first hidden layer

            For layers l = 2, ..., depth:
                self.W[l] : (width, width)
                self.b[l] : (width,)

            self.Theta : (m, width)       # weights of linear readout
            self.c     : (m,)             # biases of linear readout
        """
        # Store basic architecture parameters
        self.n = n          # input dimension
        self.m = m          # output dimension (for readout)
        self.width = width  # hidden layer width / feature dimension
        self.depth = depth  # number of hidden layers

        # Use NumPy's Generator API for reproducible random numbers.
        # This is preferable to np.random.seed + np.random.randn in new code.
        rng = np.random.default_rng(seed)

        # Lists to hold weight matrices and bias vectors for each hidden layer.
        # self.W[i] corresponds to the weight matrix of layer i,
        # self.b[i] corresponds to the bias vector of layer i.
        self.W = []
        self.b = []

        # ----- First hidden layer -----
        # Weight matrix: (width, n)
        #   - maps input x ∈ Rⁿ to hidden activation a₁ ∈ R^{width}
        self.W.append(rng.standard_normal((width, n)))

        # Bias vector: (width,)
        #   - adds a learned offset to each neuron in the first hidden layer
        self.b.append(rng.standard_normal(width))

        # ----- Remaining hidden layers (2 .. depth) -----
        # Each subsequent hidden layer maps from R^{width} to R^{width},
        # keeping the same layer width throughout the hidden stack.
        for _ in range(1, depth):
            # Weight matrix: (width, width)
            self.W.append(rng.standard_normal((width, width)))

            # Bias vector: (width,)
            self.b.append(rng.standard_normal(width))

        # ----- Final linear readout -----
        # Θ ∈ R^{m×width} maps feature vector Φ(x) ∈ R^{width} to output y(x) ∈ Rᵐ.
        self.Theta = rng.standard_normal((m, width))

        # Bias c ∈ Rᵐ is added to the linear readout.
        self.c = rng.standard_normal(m)

    @staticmethod
    def relu(z):
        """
        Apply the ReLU activation elementwise.

        Parameters
        ----------
        z : np.ndarray
            Input array of any shape.

        Returns
        -------
        np.ndarray
            Array of the same shape as z, with ReLU applied elementwise:

                ReLU(z) = max(0, z)

        Notes
        -----
        This is implemented as np.maximum(0.0, z), which is vectorized and
        efficient for NumPy arrays.
        """
        return np.maximum(0.0, z)

    def Phi(self, X):
        """
        Compute the feature map Φ(X) for a batch of inputs.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape:
                - (N, n) : batch of N inputs, each of dimension n
                - (n,)   : single input, which will be reshaped to (1, n)

        Returns
        -------
        np.ndarray
            Feature matrix Φ(X) with shape (N, width):

                - Each row i corresponds to Φ(xᵢ)ᵀ for the i-th input sample.
                - The feature dimension equals self.width.

        Notes
        -----
        Internally, this performs a forward pass through all hidden layers:

            A₀ = X
            For l = 1, ..., depth:
                Z_l = A_{l-1} W_lᵀ + b_l
                A_l = ReLU(Z_l)

            Φ(X) = A_depth

        where A_l has shape (N, width) for all hidden layers.
        """
        # Convert input to a NumPy array to ensure consistent handling.
        X = np.asarray(X)

        # If a single vector x ∈ Rⁿ is passed, shape is (n,).
        # Reshape to (1, n) so that we can treat it as a batch of size 1.
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # A will hold the activations as we propagate through layers.
        # Initial activation is simply the input data.
        # Shape: (N, n) initially.
        A = X

        # Iterate over each hidden layer's weights and biases.
        # After the first iteration, A will always have shape (N, width).
        for W, b in zip(self.W, self.b):
            # Compute affine transformation:
            #   A @ W.T has shape (N, width) because:
            #       A: (N, in_dim), W: (width, in_dim)
            #   b has shape (width,), NumPy broadcasts it across rows.
            Z = A @ W.T + b

            # Apply ReLU activation elementwise.
            A = self.relu(Z)

        # At this point A is the output of the last hidden layer, i.e., Φ(X).
        # Shape: (N, width)
        return A

    def forward(self, X):
        """
        Compute the full network output y(X) = Θ Φ(X) + c.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape:
                - (N, n) : batch of N inputs
                - (n,)   : single input, reshaped internally to (1, n)

        Returns
        -------
        np.ndarray
            Output array y(X) with shape (N, m):

                - For each input xᵢ, the corresponding output is y(xᵢ) ∈ Rᵐ.
                - If X was one-dimensional (n,), the result will still be (1, m).

        Notes
        -----
        This function is essentially a convenience wrapper that:

            1. Calls self.Phi(X) to get Φ(X) ∈ R^{N×width}
            2. Applies the linear readout:

                   y(X) = Φ(X) Θᵀ + c

           where Θ ∈ R^{m×width}, so Φ(X) Θᵀ has shape (N, m),
           and c ∈ Rᵐ is broadcast across the batch dimension.
        """
        # Compute feature matrix Φ(X) with shape (N, width)
        PhiX = self.Phi(X)

        # Apply linear readout:
        #   PhiX @ Theta.T : (N, width) @ (width, m) → (N, m)
        #   + self.c       : (m,) broadcast to each row → (N, m)
        return PhiX @ self.Theta.T + self.c