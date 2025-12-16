from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np


InitialConditionType = Literal["sin_pi", "constant"]


@dataclass
class SimulationConfig:
    """Configuration for the 1D heat equation simulation.

    Parameters
    ----------
    length : float
        Length of the rod (domain size).
    nx : int
        Number of grid points in space (including boundaries).
    total_time : float
        Total simulation time.
    dt : float
        Time step size.
    alpha : float
        Thermal diffusivity.
    left_bc : float
        Dirichlet boundary value at the left end.
    right_bc : float
        Dirichlet boundary value at the right end.
    initial_condition : {"sin_pi", "constant"}
        Type of initial condition to use.
    output_csv : str
        Path to the CSV file where the final temperature distribution is saved.
    output_plot : str
        Path to the image file where the final temperature is plotted.
    """

    length: float
    nx: int
    total_time: float
    dt: float
    alpha: float
    left_bc: float
    right_bc: float
    initial_condition: InitialConditionType
    output_csv: str
    output_plot: str

    @property
    def dx(self) -> float:
        """Spatial grid spacing."""
        return self.length / (self.nx - 1)

    @property
    def nsteps(self) -> int:
        """Number of time steps."""
        return int(self.total_time / self.dt)


def load_config(path: str) -> SimulationConfig:
    """Load simulation configuration from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON configuration file.

    Returns
    -------
    SimulationConfig
        Parsed configuration object.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If required fields are missing or invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = [
        "length",
        "nx",
        "total_time",
        "dt",
        "alpha",
        "left_bc",
        "right_bc",
        "initial_condition",
        "output_csv",
        "output_plot",
    ]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required configuration key: {key}")

    cfg = SimulationConfig(
        length=float(data["length"]),
        nx=int(data["nx"]),
        total_time=float(data["total_time"]),
        dt=float(data["dt"]),
        alpha=float(data["alpha"]),
        left_bc=float(data["left_bc"]),
        right_bc=float(data["right_bc"]),
        initial_condition=data["initial_condition"],
        output_csv=str(data["output_csv"]),
        output_plot=str(data["output_plot"]),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: SimulationConfig) -> None:
    """Validate configuration values.

    Parameters
    ----------
    cfg : SimulationConfig
        Configuration to validate.

    Raises
    ------
    ValueError
        If any parameter is invalid or violates stability conditions.
    """
    if cfg.length <= 0:
        raise ValueError("length must be positive")
    if cfg.nx < 3:
        raise ValueError("nx must be at least 3 (including boundaries)")
    if cfg.total_time <= 0:
        raise ValueError("total_time must be positive")
    if cfg.dt <= 0:
        raise ValueError("dt must be positive")
    if cfg.alpha <= 0:
        raise ValueError("alpha must be positive")
    if cfg.initial_condition not in ("sin_pi", "constant"):
        raise ValueError("initial_condition must be 'sin_pi' or 'constant'")

    # CFL-like stability condition for explicit FTCS scheme: r = alpha*dt/dx^2 <= 0.5
    r = cfg.alpha * cfg.dt / (cfg.dx**2)
    if r > 0.5:
        raise ValueError(
            f"Unstable time step: alpha*dt/dx^2 = {r:.3f} > 0.5. "
            "Reduce dt or increase nx."
        )


def initial_temperature(cfg: SimulationConfig) -> np.ndarray:
    """Construct the initial temperature distribution.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation configuration.

    Returns
    -------
    np.ndarray
        Array of shape (nx,) with initial temperatures.
    """
    x = np.linspace(0.0, cfg.length, cfg.nx)

    if cfg.initial_condition == "sin_pi":
        # u(x, 0) = sin(pi * x / L), with boundary values forced afterward
        u0 = np.sin(np.pi * x / cfg.length)
    elif cfg.initial_condition == "constant":
        # Use a constant value equal to the left boundary.
        # For this simple case we assume left_bc == right_bc.
        u0 = np.full_like(x, cfg.left_bc, dtype=float)
    else:
        raise ValueError(f"Unsupported initial condition: {cfg.initial_condition}")

    u0[0] = cfg.left_bc
    u0[-1] = cfg.right_bc
    return u0


def step_ftcs(u: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Advance the solution by one time step using FTCS scheme.

    Parameters
    ----------
    u : np.ndarray
        Current temperature distribution, shape (nx,).
    cfg : SimulationConfig
        Simulation configuration.

    Returns
    -------
    np.ndarray
        New temperature distribution after one time step, shape (nx,).
    """
    r = cfg.alpha * cfg.dt / (cfg.dx**2)
    u_new = u.copy()

    # Update interior points: i = 1..nx-2
    u_new[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])

    # Enforce Dirichlet boundary conditions
    u_new[0] = cfg.left_bc
    u_new[-1] = cfg.right_bc
    return u_new


def run_simulation(cfg: SimulationConfig) -> np.ndarray:
    """Run the full simulation.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation configuration.

    Returns
    -------
    np.ndarray
        Final temperature distribution, shape (nx,).
    """
    u = initial_temperature(cfg)
    for _ in range(cfg.nsteps):
        u = step_ftcs(u, cfg)
    return u
