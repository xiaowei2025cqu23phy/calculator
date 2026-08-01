import numpy as np
import pytest

from core import evaluate_expression, matrix_add, parse_expression, plot_function


def test_evaluate_expression_returns_float_for_real_expression():
    assert evaluate_expression("sin(pi / 2) + 1") == pytest.approx(2.0)


def test_matrix_add_accepts_semicolon_separated_input():
    result = matrix_add("1,2;3,4", "5,6;7,8")

    np.testing.assert_allclose(result, np.array([[6, 8], [10, 12]], dtype=np.complex128))


def test_plot_function_supports_constant_expressions():
    fig = plot_function("2", a=-1, b=1, points=5, show=False)
    ax = fig.axes[0]

    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.full(5, 2.0))


def test_parse_matrix_rejects_empty_text():
    with pytest.raises(ValueError, match="解析矩阵失败"):
        matrix_add("", "[[1]]")


def test_parse_expression_treats_caret_as_exponentiation():
    assert evaluate_expression("2^3") == pytest.approx(8.0)


def test_parse_expression_rejects_dunder_names():
    with pytest.raises(ValueError, match="不允许"):
        parse_expression('__import__("os")')
