import numpy as np


class ValueFunction:
    def __init__(self, T: int, ex_space: np.ndarray, ey_space: np.ndarray, etheta_space: np.ndarray):
        self.T = T
        self.ex_space = ex_space
        self.ey_space = ey_space
        self.etheta_space = etheta_space
        self.value = np.zeros((T, len(ex_space), len(ey_space), len(etheta_space)))

    def copy_from(self, other: "ValueFunction"):
        """
        Update the underlying value function storage with another value function
        """
        self.value = other.value.copy()

    def update(self, t: int, ex_id: int, ey_id: int, etheta_id: int, target_value: float):
        """
        Update the value function at given states
        Args:
            t: time step
            ex_id: x position error index
            ey_id: y position error index
            etheta_id: theta error index
            target_value: target value
        """
        self.value[t, ex_id, ey_id, etheta_id] = target_value

    def __call__(self, t: int, ex_id: int, ey_id: int, etheta_id: int) -> float:
        """
        Get the value function results at given states
        Args:
            t: time step
            ex_id: x position error index
            ey_id: y position error index
            etheta_id: theta error index
        Returns:
            value function results
        """
        return self.value[t, ex_id, ey_id, etheta_id]

    def copy(self):
        """
        Create a copy of the value function
        Returns:
            a copy of the value function
        """
        copied_value = ValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space)
        copied_value.value = self.value.copy()
        return copied_value


class GridValueFunction(ValueFunction):
    """
    Grid-based value function
    """
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        super().__init__(T, ex_space, ey_space, etheta_space)


class FeatureValueFunction(ValueFunction):
    """
    Feature-based value function
    """
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        super().__init__(T, ex_space, ey_space, etheta_space)
