# Contributing Guidelines

Thank you for your interest in contributing to the Titanic ML Pipeline project!

## Getting Started

1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/YOUR_USERNAME/mlops-pipeline.git`
3. **Create a feature branch**: `git checkout -b feature/your-feature-name`
4. **Install development dependencies**: `pip install -r requirements.txt && pip install pytest pytest-cov black flake8`

## Development Workflow

### Code Style

We follow PEP 8 conventions. Format your code:

```bash
black src/ scripts/ config/
```

Lint your code:

```bash
flake8 src/ scripts/ --max-line-length=100
```

### Writing Tests

Create tests in a `tests/` directory:

```bash
mkdir tests
touch tests/test_data_processing.py
```

Example test:

```python
import unittest
from src.data_processing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
    
    def test_load_data(self):
        df = self.processor.load_data('titanic.csv')
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()
```

Run tests:

```bash
pytest tests/ -v --cov=src
```

## Commit Guidelines

Follow conventional commits:

```
feat: Add feature description
fix: Fix bug description
docs: Update documentation
test: Add tests
refactor: Refactor code
perf: Performance improvement
```

Example:

```bash
git commit -m "feat: Add cross-validation to model training"
```

## Pull Request Process

1. **Update** CHANGELOG.md with your changes
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v
   flake8 src/ scripts/ --max-line-length=100
   ```
5. **Push** to your fork
6. **Create** a Pull Request with:
   - Clear title describing changes
   - Description of what changed and why
   - Reference any related issues (e.g., "Fixes #123")

## Code Review Process

- Maintainers will review your PR
- Address feedback and update your PR
- Once approved, your changes will be merged

## Areas for Contribution

- 🐛 **Bug fixes**: Report issues and submit fixes
- ✨ **Features**: Add new capabilities (hyperparameter tuning, new models, etc.)
- 📚 **Documentation**: Improve guides and examples
- ⚡ **Performance**: Optimize code and infrastructure
- 🧪 **Tests**: Increase test coverage
- 🔒 **Security**: Report vulnerabilities responsibly

## Development Tips

### Local Training

```bash
python src/train.py --data titanic.csv --output ./test_model --max-iter 100
```

### Testing API

```bash
python scripts/app.py
# In another terminal:
curl http://localhost:5000/health
```

### Debugging

Add logging to your code:

```python
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")
```

## Release Process

1. Update version in `config/config.py`
2. Update CHANGELOG.md
3. Create annotated tag: `git tag -a v1.1.0 -m "Release v1.1.0"`
4. Push tag: `git push origin v1.1.0`

## Questions?

- Check existing issues and discussions
- Open a new issue for questions
- Contact maintainers directly

## Code of Conduct

- Be respectful and inclusive
- Welcome diverse perspectives
- Address conflicts professionally

Thank you for contributing! 🙌
