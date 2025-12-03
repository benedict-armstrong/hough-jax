import numpy as np
import jax.numpy as jnp
import jax
import pytest
from skimage.transform import probabilistic_hough_line as skimage_hough
from src.probabilistic_hough_line import probabilistic_hough_line as jax_hough
from src.probabilistic_hough_line import _probabilistic_hough_line_impl


def lines_match(jax_lines, skimage_lines, tolerance=1):
    """Check if two sets of lines match within tolerance."""
    if len(jax_lines) != len(skimage_lines):
        return False

    # Convert to sets of sorted tuples for comparison
    def normalize_line(line):
        (x0, y0), (x1, y1) = line
        # Sort endpoints to make comparison order-independent
        if (x0, y0) > (x1, y1):
            return ((x1, y1), (x0, y0))
        return ((x0, y0), (x1, y1))

    jax_set = set(normalize_line(l) for l in jax_lines)
    skimage_set = set(normalize_line(l) for l in skimage_lines)

    return jax_set == skimage_set


def create_test_image_horizontal_line():
    """Create a simple image with a horizontal line."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50, 20:80] = 1
    return img


def create_test_image_vertical_line():
    """Create a simple image with a vertical line."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 50] = 1
    return img


def create_test_image_diagonal_line():
    """Create a simple image with a diagonal line."""
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(60):
        img[20 + i, 20 + i] = 1
    return img


def create_test_image_multiple_lines():
    """Create an image with multiple lines."""
    img = np.zeros((100, 100), dtype=np.uint8)
    # Horizontal line
    img[30, 10:90] = 1
    # Vertical line
    img[10:90, 70] = 1
    # Diagonal line
    for i in range(50):
        img[40 + i, 10 + i] = 1
    return img


class TestProbabilisticHoughLine:
    """Tests for JAX probabilistic Hough line transform."""

    def test_horizontal_line(self):
        """Test detection of a horizontal line."""
        img = create_test_image_horizontal_line()
        seed = 42

        # Run skimage version
        rng_np = np.random.default_rng(seed)
        skimage_lines = skimage_hough(img, threshold=10, line_length=30, line_gap=3, rng=rng_np)

        # Run JAX version
        rng_jax = jax.random.PRNGKey(seed)
        jax_lines = jax_hough(
            jnp.array(img), threshold=10, line_length=30, line_gap=3, rng=rng_jax
        )

        assert len(jax_lines) == len(skimage_lines), (
            f"Expected {len(skimage_lines)} lines, got {len(jax_lines)}"
        )
        assert lines_match(jax_lines, skimage_lines)

    def test_vertical_line(self):
        """Test detection of a vertical line."""
        img = create_test_image_vertical_line()
        seed = 42

        rng_np = np.random.default_rng(seed)
        skimage_lines = skimage_hough(img, threshold=10, line_length=30, line_gap=3, rng=rng_np)

        rng_jax = jax.random.PRNGKey(seed)
        jax_lines = jax_hough(
            jnp.array(img), threshold=10, line_length=30, line_gap=3, rng=rng_jax
        )

        assert len(jax_lines) == len(skimage_lines)
        assert lines_match(jax_lines, skimage_lines)

    def test_diagonal_line(self):
        """Test detection of a diagonal line."""
        img = create_test_image_diagonal_line()
        seed = 42

        rng_np = np.random.default_rng(seed)
        skimage_lines = skimage_hough(img, threshold=10, line_length=30, line_gap=3, rng=rng_np)

        rng_jax = jax.random.PRNGKey(seed)
        jax_lines = jax_hough(
            jnp.array(img), threshold=10, line_length=30, line_gap=3, rng=rng_jax
        )

        assert len(jax_lines) == len(skimage_lines)
        assert lines_match(jax_lines, skimage_lines)

    def test_multiple_lines(self):
        """Test detection of multiple lines."""
        img = create_test_image_multiple_lines()
        seed = 42

        rng_np = np.random.default_rng(seed)
        skimage_lines = skimage_hough(img, threshold=10, line_length=30, line_gap=3, rng=rng_np)

        rng_jax = jax.random.PRNGKey(seed)
        jax_lines = jax_hough(
            jnp.array(img), threshold=10, line_length=30, line_gap=3, rng=rng_jax
        )

        assert len(jax_lines) == len(skimage_lines)
        assert lines_match(jax_lines, skimage_lines)

    def test_empty_image(self):
        """Test with an empty image (no edges)."""
        img = np.zeros((50, 50), dtype=np.uint8)

        rng_np = np.random.default_rng(42)
        skimage_lines = skimage_hough(img, rng=rng_np)

        rng_jax = jax.random.PRNGKey(42)
        jax_lines = jax_hough(jnp.array(img), rng=rng_jax)

        assert len(jax_lines) == 0
        assert len(skimage_lines) == 0

    def test_default_theta(self):
        """Test that default theta values work correctly."""
        img = create_test_image_horizontal_line()
        seed = 42

        rng_np = np.random.default_rng(seed)
        skimage_lines = skimage_hough(img, threshold=10, line_length=30, line_gap=3, rng=rng_np)

        rng_jax = jax.random.PRNGKey(seed)
        jax_lines = jax_hough(
            jnp.array(img), threshold=10, line_length=30, line_gap=3, rng=rng_jax
        )

        assert len(jax_lines) == len(skimage_lines)

    def test_custom_theta(self):
        """Test with custom theta values."""
        img = create_test_image_horizontal_line()
        seed = 42
        theta = np.linspace(-np.pi / 2, np.pi / 2, 90, endpoint=False)

        rng_np = np.random.default_rng(seed)
        skimage_lines = skimage_hough(
            img, threshold=10, line_length=30, line_gap=3, theta=theta, rng=rng_np
        )

        rng_jax = jax.random.PRNGKey(seed)
        jax_lines = jax_hough(
            jnp.array(img),
            threshold=10,
            line_length=30,
            line_gap=3,
            theta=jnp.array(theta),
            rng=rng_jax,
        )

        assert len(jax_lines) == len(skimage_lines)
        assert lines_match(jax_lines, skimage_lines)

    def test_different_thresholds(self):
        """Test with different threshold values."""
        img = create_test_image_multiple_lines()
        seed = 42

        for threshold in [5, 10, 20]:
            rng_np = np.random.default_rng(seed)
            skimage_lines = skimage_hough(
                img, threshold=threshold, line_length=30, line_gap=3, rng=rng_np
            )

            rng_jax = jax.random.PRNGKey(seed)
            jax_lines = jax_hough(
                jnp.array(img),
                threshold=threshold,
                line_length=30,
                line_gap=3,
                rng=rng_jax,
            )

            assert len(jax_lines) == len(skimage_lines), (
                f"Threshold {threshold}: expected {len(skimage_lines)}, got {len(jax_lines)}"
            )

    def test_different_line_lengths(self):
        """Test with different minimum line length values."""
        img = create_test_image_horizontal_line()
        seed = 42

        for line_length in [20, 40, 60]:
            rng_np = np.random.default_rng(seed)
            skimage_lines = skimage_hough(
                img, threshold=10, line_length=line_length, line_gap=3, rng=rng_np
            )

            rng_jax = jax.random.PRNGKey(seed)
            jax_lines = jax_hough(
                jnp.array(img),
                threshold=10,
                line_length=line_length,
                line_gap=3,
                rng=rng_jax,
            )

            assert len(jax_lines) == len(skimage_lines), (
                f"Line length {line_length}: expected {len(skimage_lines)}, got {len(jax_lines)}"
            )

    def test_output_format(self):
        """Test that output format matches skimage format."""
        img = create_test_image_horizontal_line()

        rng_jax = jax.random.PRNGKey(42)
        jax_lines = jax_hough(
            jnp.array(img), threshold=10, line_length=30, line_gap=3, rng=rng_jax
        )

        # Check output format: list of ((x0, y0), (x1, y1))
        assert isinstance(jax_lines, list)
        if len(jax_lines) > 0:
            line = jax_lines[0]
            assert len(line) == 2  # Two endpoints
            assert len(line[0]) == 2  # (x0, y0)
            assert len(line[1]) == 2  # (x1, y1)


def jit_result_to_lines(lines_result, nlines_result):
    """Convert JIT output to list of line tuples."""
    nlines_int = int(nlines_result)
    lines_np = np.array(lines_result[:nlines_int])
    return [
        ((int(line[0, 0]), int(line[0, 1])), (int(line[1, 0]), int(line[1, 1])))
        for line in lines_np
    ]


class TestProbabilisticHoughLineJIT:
    """Tests for JIT-compiled JAX probabilistic Hough line transform."""

    @pytest.fixture
    def jit_hough_impl(self):
        """Create a JIT-compiled version of the Hough transform implementation."""
        return jax.jit(
            _probabilistic_hough_line_impl,
            static_argnames=["threshold", "line_length", "line_gap", "height", "width"],
        )

    def test_jit_horizontal_line(self, jit_hough_impl):
        """Test JIT-compiled detection of a horizontal line."""
        img = create_test_image_horizontal_line()
        theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 180, endpoint=False)
        rng = jax.random.PRNGKey(42)
        height, width = img.shape

        # Run eager version
        eager_lines = jax_hough(jnp.array(img), threshold=10, line_length=30, line_gap=3, rng=rng)

        # Run JIT version
        lines_result, nlines_result = jit_hough_impl(
            jnp.array(img), 10, 30, 3, theta, rng, height, width
        )
        jit_lines = jit_result_to_lines(lines_result, nlines_result)

        assert len(jit_lines) == len(eager_lines)
        assert lines_match(jit_lines, eager_lines)

    def test_jit_multiple_lines(self, jit_hough_impl):
        """Test JIT-compiled detection of multiple lines."""
        img = create_test_image_multiple_lines()
        theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 180, endpoint=False)
        rng = jax.random.PRNGKey(42)
        height, width = img.shape

        eager_lines = jax_hough(jnp.array(img), threshold=10, line_length=30, line_gap=3, rng=rng)

        lines_result, nlines_result = jit_hough_impl(
            jnp.array(img), 10, 30, 3, theta, rng, height, width
        )
        jit_lines = jit_result_to_lines(lines_result, nlines_result)

        assert len(jit_lines) == len(eager_lines)
        assert lines_match(jit_lines, eager_lines)

    def test_jit_recompilation_different_params(self, jit_hough_impl):
        """Test JIT recompilation with different static parameters."""
        img = create_test_image_horizontal_line()
        theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 180, endpoint=False)
        rng = jax.random.PRNGKey(42)
        height, width = img.shape

        # First call with one set of params
        lines1_result, nlines1 = jit_hough_impl(
            jnp.array(img), 10, 30, 3, theta, rng, height, width
        )
        lines1 = jit_result_to_lines(lines1_result, nlines1)

        # Second call with different params (triggers recompilation)
        lines2_result, nlines2 = jit_hough_impl(
            jnp.array(img), 5, 20, 5, theta, rng, height, width
        )
        lines2 = jit_result_to_lines(lines2_result, nlines2)

        # Both should produce valid results
        assert isinstance(lines1, list)
        assert isinstance(lines2, list)

    def test_jit_different_image_sizes(self, jit_hough_impl):
        """Test JIT with different image sizes."""
        theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 180, endpoint=False)
        rng = jax.random.PRNGKey(42)

        # Small image
        img_small = np.zeros((50, 50), dtype=np.uint8)
        img_small[25, 10:40] = 1
        lines_result, nlines = jit_hough_impl(jnp.array(img_small), 10, 20, 3, theta, rng, 50, 50)
        lines_small = jit_result_to_lines(lines_result, nlines)

        # Larger image
        img_large = np.zeros((150, 150), dtype=np.uint8)
        img_large[75, 20:130] = 1
        lines_result, nlines = jit_hough_impl(
            jnp.array(img_large), 10, 50, 3, theta, rng, 150, 150
        )
        lines_large = jit_result_to_lines(lines_result, nlines)

        assert len(lines_small) >= 0
        assert len(lines_large) >= 0

    def test_jit_consistency_multiple_calls(self, jit_hough_impl):
        """Test that JIT produces consistent results across multiple calls."""
        img = create_test_image_horizontal_line()
        theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 180, endpoint=False)
        rng = jax.random.PRNGKey(42)
        height, width = img.shape

        # Run multiple times with same inputs
        results = []
        for _ in range(3):
            lines_result, nlines = jit_hough_impl(
                jnp.array(img), 10, 30, 3, theta, rng, height, width
            )
            lines = jit_result_to_lines(lines_result, nlines)
            results.append(lines)

        # All results should be identical
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0])
            assert lines_match(results[i], results[0])

    def test_jit_vs_skimage(self, jit_hough_impl):
        """Test JIT-compiled version against skimage reference."""
        img = create_test_image_diagonal_line()
        seed = 42
        theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 180, endpoint=False)
        height, width = img.shape

        rng_np = np.random.default_rng(seed)
        skimage_lines = skimage_hough(img, threshold=10, line_length=30, line_gap=3, rng=rng_np)

        rng_jax = jax.random.PRNGKey(seed)
        lines_result, nlines = jit_hough_impl(
            jnp.array(img), 10, 30, 3, theta, rng_jax, height, width
        )
        jit_lines = jit_result_to_lines(lines_result, nlines)

        assert len(jit_lines) == len(skimage_lines)
        assert lines_match(jit_lines, skimage_lines)

    def test_jit_empty_image(self, jit_hough_impl):
        """Test JIT with empty image."""
        img = np.zeros((50, 50), dtype=np.uint8)
        theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 180, endpoint=False)
        rng = jax.random.PRNGKey(42)

        lines_result, nlines = jit_hough_impl(jnp.array(img), 10, 30, 3, theta, rng, 50, 50)
        lines = jit_result_to_lines(lines_result, nlines)

        assert len(lines) == 0

    def test_jit_custom_theta(self, jit_hough_impl):
        """Test JIT with custom theta values."""
        img = create_test_image_horizontal_line()
        theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 90, endpoint=False)
        rng = jax.random.PRNGKey(42)
        height, width = img.shape

        eager_lines = jax_hough(
            jnp.array(img),
            threshold=10,
            line_length=30,
            line_gap=3,
            theta=theta,
            rng=rng,
        )

        lines_result, nlines = jit_hough_impl(jnp.array(img), 10, 30, 3, theta, rng, height, width)
        jit_lines = jit_result_to_lines(lines_result, nlines)

        assert len(jit_lines) == len(eager_lines)
        assert lines_match(jit_lines, eager_lines)
