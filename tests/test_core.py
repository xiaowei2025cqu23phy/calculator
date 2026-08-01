import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import evaluate_expression, matrix_add, plot_function


def test_evaluate_expression_returns_float_for_real_expression():
    assert evaluate_expression("sin(pi / 2) + 1") == pytest.approx(2.0)


def test_matrix_add_accepts_semicolon_separated_input():
    result = matrix_add("1,2;3,4", "5,6;7,8")

    np.testing.assert_allclose(result, np.array([[6, 8], [10, 12]], dtype=np.complex128))


def test_plot_function_supports_constant_expressions():
    fig = plot_function("2", a=-1, b=1, points=5, show=False)
    ax = fig.axes[0]

    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.full(5, 2.0))
