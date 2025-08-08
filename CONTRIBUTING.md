# Contributing to Godly AI

Thank you for your interest in contributing to the Godly AI project! This document provides guidelines and information for contributors.

## 🎯 Project Vision

We're building the world's first fully autonomous, self-learning AI system using neuromorphic computing principles. Our goal is to create an AI that learns without human supervision and outperforms existing models on agentic tasks.

## 🤝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Code contributions**: Core system implementation, bug fixes, optimizations
- **Research contributions**: Neuromorphic algorithms, learning mechanisms, benchmarks
- **Documentation**: API docs, tutorials, architecture explanations
- **Testing**: Unit tests, integration tests, performance benchmarks
- **Bug reports**: Issue identification and reproduction steps
- **Feature requests**: New capabilities and improvements

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Godly-AI.git
   cd Godly-AI
   ```
3. **Set up the development environment**:
   ```bash
   poetry install
   poetry shell
   pre-commit install
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Workflow

1. **Make your changes** following our coding standards
2. **Add tests** for new functionality
3. **Run the test suite**:
   ```bash
   pytest tests/
   ```
4. **Run code quality checks**:
   ```bash
   pre-commit run --all-files
   ```
5. **Commit your changes**:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request** on GitHub

## 📝 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black formatter default)
- **Import organization**: Use isort with Black profile
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public APIs

### Code Quality Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting and style checking
- **mypy**: Static type checking
- **pre-commit**: Automated checks before commits

### Neuromorphic Code Guidelines

When working with neuromorphic components:

- **Use JAX**: Prefer JAX over NumPy for performance-critical code
- **Vectorization**: Write vectorized operations when possible
- **Memory efficiency**: Consider memory usage in spike processing
- **Biological plausibility**: Maintain biological realism where applicable
- **Documentation**: Explain neuromorphic concepts clearly

## 🧪 Testing Guidelines

### Test Categories

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark speed and memory usage
- **Validation tests**: Verify biological plausibility

### Writing Tests

```python
import pytest
import jax.numpy as jnp
from src.core.neurons import SpikingNeuron

class TestSpikingNeuron:
    def test_integrate_and_fire_dynamics(self):
        """Test basic integrate-and-fire behavior."""
        neuron = SpikingNeuron(threshold=1.0, reset_potential=0.0)
        
        # Test sub-threshold integration
        spike = neuron.integrate_and_fire(0.5, dt=1.0)
        assert not spike
        
        # Test supra-threshold firing
        spike = neuron.integrate_and_fire(1.5, dt=1.0)
        assert spike
    
    @pytest.mark.benchmark
    def test_neuron_performance(self, benchmark):
        """Benchmark neuron computation speed."""
        neuron = SpikingNeuron(threshold=1.0, reset_potential=0.0)
        input_current = jnp.array([0.5, 1.5, 0.3, 2.0])
        
        result = benchmark(neuron.process_batch, input_current)
        assert len(result) == len(input_current)
```

### Test Coverage

- Aim for >95% code coverage
- Include edge cases and error conditions
- Test both correctness and performance
- Validate against biological data where applicable

## 📚 Documentation Standards

### Code Documentation

- **Docstrings**: All public functions, classes, and modules
- **Type hints**: Complete type annotations
- **Examples**: Include usage examples in docstrings
- **References**: Cite relevant research papers

### API Documentation

```python
def integrate_and_fire(
    self, 
    input_current: float, 
    dt: float = 1.0
) -> bool:
    """Integrate input current and fire if threshold is exceeded.
    
    This implements the leaky integrate-and-fire neuron model with
    exponential decay of membrane potential.
    
    Args:
        input_current: Input current in arbitrary units
        dt: Time step in milliseconds
        
    Returns:
        True if neuron fires (spikes), False otherwise
        
    Example:
        >>> neuron = SpikingNeuron(threshold=1.0)
        >>> spike = neuron.integrate_and_fire(1.5)
        >>> print(spike)  # True
        
    References:
        Gerstner, W., & Kistler, W. M. (2002). Spiking neuron models.
    """
```

## 🔬 Research Contributions

### Neuromorphic Algorithms

When contributing neuromorphic algorithms:

- **Literature review**: Reference existing research
- **Biological basis**: Explain biological inspiration
- **Mathematical formulation**: Provide clear equations
- **Implementation**: Efficient JAX implementation
- **Validation**: Compare with biological data

### Performance Benchmarks

- **Standardized metrics**: Use consistent measurement methods
- **Hardware specifications**: Document test hardware
- **Reproducibility**: Provide complete benchmark code
- **Comparison**: Compare with existing methods

## 🐛 Bug Reports

### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- JAX version: [e.g., 0.4.13]
- Hardware: [e.g., Intel i7-12700K, 32GB RAM]

**Additional Context**
Any other relevant information.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of the proposed feature.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Implementation**
How should this feature be implemented?

**Alternatives Considered**
What alternative solutions have you considered?

**Additional Context**
Any other relevant information or examples.
```

## 🏆 Recognition

Contributors will be recognized in:

- **README.md**: Major contributors listed
- **CHANGELOG.md**: Contributions noted in releases
- **Research papers**: Co-authorship for significant research contributions
- **Conference presentations**: Acknowledgment in talks

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: team@godly-ai.com for private inquiries

## 📄 License

By contributing to Godly AI, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Thank you for contributing to the future of neuromorphic AI! Your contributions help advance the field and bring us closer to truly autonomous artificial intelligence.