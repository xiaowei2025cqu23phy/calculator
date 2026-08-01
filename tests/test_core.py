import numpy as np
import pytest

from core import definite_integral, evaluate_expression, matrix_add, matrix_mul, parse_expression, plot_function, split_expressions


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


def test_parse_expression_rejects_unknown_functions():
    with pytest.raises(ValueError, match="不支持的函数"):
        parse_expression("unknown_func(1)")


def test_definite_integral_of_linear_function():
    value, error = definite_integral("x", "x", 0, 1)

    assert value == pytest.approx(0.5)
    assert error >= 0


def test_plot_function_draws_real_and_imaginary_parts():
    fig = plot_function("I*x", a=0, b=1, points=4, show=False)
    ax = fig.axes[0]

    assert len(ax.lines) == 2
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.zeros(4))
    np.testing.assert_allclose(ax.lines[1].get_ydata(), np.linspace(0, 1, 4))


def test_matrix_mul_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="内维度"):
        matrix_mul("[[1, 2]]", "[[1, 2]]")


def test_split_expressions_handles_semicolons_and_newlines():
    assert split_expressions("sin(x); cos(x)\n tan(x); ; 2") == [
        "sin(x)",
        "cos(x)",
        "tan(x)",
        "2",
    ]
