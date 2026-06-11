import math
import numpy as np

class M_FunnelTwoLayerReLuMRAC:
  @staticmethod
  def computeControlLawTwoLayerReLu(
    K_hat_x: np.ndarray,
    x: np.ndarray,
    K_hat_r: np.ndarray,
    r: np.ndarray,
    Theta_hat: np.ndarray,
    Phi: np.ndarray,
    K_hat_g: np.ndarray,
    e: np.ndarray,
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
      + K_hat_g.T * e
      - Theta_hat_relu.T * Phi_relu
    )

    return control_input
  
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
  def computeAllRobustAdaptiveLawsTwoLayerReLu(
    Gamma_x: np.ndarray,
    x: np.ndarray,
    Gamma_r: np.ndarray,
    r: np.ndarray,
    Gamma_Theta: np.ndarray,
    Phi_regressor_vector: np.ndarray,
    Gamma_g: np.ndarray,
    e: np.ndarray,
    Gamma_Theta_ReLu: np.ndarray,
    Phi_regressor_NN: np.ndarray,
    eTranspose_P_B: np.ndarray,
    dead_zone_value: float,
    sigma_x: float,
    sigma_r: float,
    sigma_Theta: float,
    sigma_g: float,
    sigma_Theta_ReLu: float,
    eTranspose_P_B_norm: float,
    K_hat_x: np.ndarray,
    K_hat_r: np.ndarray,
    Theta_hat: np.ndarray,
    K_hat_g: np.ndarray,
    Theta_hat_ReLu: np.ndarray,
    use_deadzone: bool,
    use_emodification: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all Two-Layer MRAC adaptive laws with OPTIONAL dead-zone and e-modification.

    Returns:
      - (K_hat_x_dot, K_hat_r_dot, Theta_hat_dot, K_hat_g_dot)
    """
    K_hat_x_dot = M_FunnelTwoLayerReLuMRAC.computeRobustAdaptiveLaw(
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

    K_hat_r_dot = M_FunnelTwoLayerReLuMRAC.computeRobustAdaptiveLaw(
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

    Theta_hat_dot = M_FunnelTwoLayerReLuMRAC.computeRobustAdaptiveLaw(
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

    K_hat_g_dot = M_FunnelTwoLayerReLuMRAC.computeRobustAdaptiveLaw(
      -Gamma_g,
      dead_zone_value,
      e,
      eTranspose_P_B,
      sigma_g,
      eTranspose_P_B_norm,
      K_hat_g,
      use_deadzone,
      use_emodification
    )

    Theta_hat_ReLu_dot = M_FunnelTwoLayerReLuMRAC.computeRobustAdaptiveLaw(
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

    return (K_hat_x_dot, K_hat_r_dot, Theta_hat_dot, K_hat_g_dot, Theta_hat_ReLu_dot)