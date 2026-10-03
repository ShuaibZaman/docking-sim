from gymnasium.envs.registration import register

register(
    id="Docking-v0",
    entry_point="docking_sim.env.docking_env:DockingEnv",
)

__all__ = ["__version__"]
__version__ = "0.1.0"
