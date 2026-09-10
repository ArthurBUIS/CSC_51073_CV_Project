import numpy as np
import pytest

from metric import cosine_sim, joint_angle, angle_similarity, frame_to_angle_vector
from ExerciseClasses import EXERCISES


def test_cosine_sim_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_sim(v, v) == pytest.approx(1.0)


def test_cosine_sim_orthogonal_vectors():
    assert cosine_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_sim_zero_vector_returns_zero():
    assert cosine_sim(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0


def test_joint_angle_right_angle():
    # B at origin, A along +x, C along +y -> angle ABC = 90 degrees
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 0.0])
    C = np.array([0.0, 1.0, 0.0])
    assert joint_angle(A, B, C) == pytest.approx(np.pi / 2)


def test_joint_angle_straight_line():
    # A and C on opposite sides of B -> angle ABC = 180 degrees
    A = np.array([-1.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 0.0])
    C = np.array([1.0, 0.0, 0.0])
    assert joint_angle(A, B, C) == pytest.approx(np.pi)


def test_frame_to_angle_vector_length_matches_triplets():
    # 33 landmarks * 3 coords, all set to a simple pattern
    frame = np.tile(np.array([0.0, 0.0, 0.0]), 33) + 0.01 * np.arange(99)
    angles = frame_to_angle_vector(frame)
    assert angles.shape == (8,)


def test_angle_similarity_decreases_as_angles_diverge():
    exercise = EXERCISES["squat"]
    zero_diff = np.zeros(8)
    small_diff = np.full(8, 0.1)
    large_diff = np.full(8, 1.0)

    score_identical = angle_similarity(zero_diff, zero_diff, exercise)
    score_small = angle_similarity(zero_diff, small_diff, exercise)
    score_large = angle_similarity(zero_diff, large_diff, exercise)

    assert score_identical >= score_small >= score_large
