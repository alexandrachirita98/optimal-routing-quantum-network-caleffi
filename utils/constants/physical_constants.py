"""
Paper's reference values (Caleffi 2017, Section V, Table 1).
"""

from utils.models.physical_params import PhysicalParams

PARAMS = PhysicalParams(
    p_ht  = 0.53,
    nu_h  = 0.80,
    nu_t  = 0.80,
    nu_o  = 0.39,
    l0    = 22_000.0,    # 22 km
    cf    = 2.0e8,       # 2 × 10⁸ m/s
    tau_p = 5.9e-6,      # 5.9 μs
    tau_h = 20.0e-6,     # 20 μs
    tau_t = 10.0e-6,     # 10 μs
    tau_d = 100.0e-6,    # 100 μs
    tau_o = 10.0e-6,     # 10 μs
    tau_a = 10.0e-6,     # 10 μs
    nu_a  = 0.39,
    t_ch  = 10.0e-3,     # 10 ms
)
