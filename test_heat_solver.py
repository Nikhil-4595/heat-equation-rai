import unittest
import numpy as np

from heat_solver import SimulationConfig, run_simulation


class TestHeatSolver(unittest.TestCase):
    def test_constant_initial_condition_remains_constant(self) -> None:
        """Constant initial condition with equal boundaries should stay constant."""
        cfg = SimulationConfig(
            length=1.0,
            nx=21,
            total_time=0.01,
            dt=0.0001,
            alpha=1.0,
            left_bc=2.0,
            right_bc=2.0,
            initial_condition="constant",
            output_csv="output/test.csv",
            output_plot="output/test.png",
        )

        u_final = run_simulation(cfg)

        # All values should be close to 2.0
        self.assertTrue(np.allclose(u_final, 2.0, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
