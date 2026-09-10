import numpy as np

from ExerciseClasses import EXERCISES, ExerciseConfig


def test_default_angle_weights_fallback_to_ones():
    config = ExerciseConfig(name="dummy", axis=1, opti=1, landmark_id=0, sensibility=0.3)
    assert np.array_equal(config.angle_weights, np.ones(8, dtype=np.float32))


def test_all_exercises_have_valid_config():
    for name, config in EXERCISES.items():
        assert config.name == name
        assert config.axis in (0, 1, 2)
        assert config.opti in (-1, 1)
        assert config.angle_weights.shape == (8,)
        assert config.a > 0
