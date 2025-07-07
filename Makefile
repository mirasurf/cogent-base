# =============================================================================
# Cogent AI Agent System - Makefile
# =============================================================================

# Configuration
PYTHON := python3
HATCH := hatch
APP_MODULE := app.main:app
HOST := 0.0.0.0
PORT := 6726
PYTHON_MODULES := src/app src/cogent
TEST_DIR := tests
DOCKER_IMAGE := cogent
DOCKER_TAG := latest

# Colors for output
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[0;33m
RED := \033[0;31m
RESET := \033[0m

# =============================================================================
# Development Commands
# =============================================================================

.PHONY: dev dev-debug install install-dev clean help

# Install dependencies
install:
	@echo "$(BLUE)📦 Installing dependencies...$(RESET)"
	@$(HATCH) env create
	@echo "$(GREEN)✅ Dependencies installed successfully!$(RESET)"

# Install development dependencies
install-dev:
	@echo "$(BLUE)🔧 Installing development dependencies...$(RESET)"
	@$(HATCH) env create
	@$(HATCH) run pip install -e ".[dev]"
	@echo "$(GREEN)✅ Development dependencies installed successfully!$(RESET)"

# =============================================================================
# Code Quality Commands
# =============================================================================

.PHONY: format format-check lint lint-fix quality

# Format code (black, isort, autoflake)
format:
	@echo "$(BLUE)🎨 Formatting code...$(RESET)"
	@$(HATCH) run autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables $(PYTHON_MODULES)
	@$(HATCH) run isort $(PYTHON_MODULES)
	@$(HATCH) run black $(PYTHON_MODULES)
	@echo "$(GREEN)✅ Code formatting complete!$(RESET)"

# Check if code is properly formatted
format-check:
	@echo "$(BLUE)🔍 Checking code formatting...$(RESET)"
	@$(HATCH) run black --check $(PYTHON_MODULES) || (echo "$(RED)❌ Code formatting check failed. Run 'make format' to fix.$(RESET)" && exit 1)
	@$(HATCH) run isort --check-only $(PYTHON_MODULES) || (echo "$(RED)❌ Import sorting check failed. Run 'make format' to fix.$(RESET)" && exit 1)
	@echo "$(GREEN)✅ Code formatting check passed!$(RESET)"

# Lint code
lint:
	@echo "$(BLUE)🔍 Running linters...$(RESET)"
	@$(HATCH) run flake8 --max-line-length=120 --extend-ignore=E203,W503 $(PYTHON_MODULES)
	@echo "$(GREEN)✅ Linting passed!$(RESET)"

# Auto-fix linting issues where possible
lint-fix:
	@echo "$(BLUE)🔧 Auto-fixing linting issues...$(RESET)"
	@$(HATCH) run autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables $(PYTHON_MODULES)
	@echo "$(GREEN)✅ Linting fixes applied!$(RESET)"

# Run all quality checks
quality: format-check lint
	@echo "$(GREEN)🎉 All quality checks passed!$(RESET)"

# =============================================================================
# Testing Commands
# =============================================================================

.PHONY: test test-unit test-integration test-coverage test-watch

# Run all tests
test:
	@echo "$(BLUE)🧪 Running all tests...$(RESET)"
	@$(HATCH) run pytest $(TEST_DIR) -v

# Run unit tests only
test-unit:
	@echo "$(BLUE)🧪 Running unit tests...$(RESET)"
	@$(HATCH) run pytest $(TEST_DIR) -v -m "not integration"

# Run integration tests only
test-integration:
	@echo "$(BLUE)🧪 Running integration tests...$(RESET)"
	@$(HATCH) run pytest $(TEST_DIR) -v -m "integration"

# Run tests with coverage
test-coverage:
	@echo "$(BLUE)🧪 Running tests with coverage...$(RESET)"
	@$(HATCH) run pytest $(TEST_DIR) --cov=$(PYTHON_MODULES) --cov-report=html --cov-report=term-missing

# Run tests in watch mode
test-watch:
	@echo "$(BLUE)👀 Running tests in watch mode...$(RESET)"
	@$(HATCH) run pytest-watch $(TEST_DIR) -- -v

# =============================================================================
# Build and Package Commands
# =============================================================================

.PHONY: build build-wheel build-sdist package

# Build package
build:
	@echo "$(BLUE)🔨 Building package...$(RESET)"
	@$(HATCH) build
	@echo "$(GREEN)✅ Package built successfully!$(RESET)"

# Build wheel only
build-wheel:
	@echo "$(BLUE)🔨 Building wheel...$(RESET)"
	@$(HATCH) build --target wheel

# Build source distribution only
build-sdist:
	@echo "$(BLUE)🔨 Building source distribution...$(RESET)"
	@$(HATCH) build --target sdist

# Package for distribution
package: clean build
	@echo "$(GREEN)📦 Package ready for distribution!$(RESET)"

# =============================================================================
# Utility Commands
# =============================================================================

.PHONY: clean clean-all shell check-env requirements

# Clean Python cache and build artifacts
clean:
	@echo "$(BLUE)🧹 Cleaning Python cache and build artifacts...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyd" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/ 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete!$(RESET)"

# Clean everything including Docker
clean-all: clean docker-clean
	@echo "$(GREEN)✅ Complete cleanup finished!$(RESET)"

# Activate development shell
shell:
	@echo "$(BLUE)🐚 Activating development shell...$(RESET)"
	@$(HATCH) shell

# Check environment
check-env:
	@echo "$(BLUE)🔍 Checking environment...$(RESET)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Hatch version: $$($(HATCH) --version)"
	@echo "Working directory: $$(pwd)"
	@echo "Python modules: $(PYTHON_MODULES)"

# Generate requirements files
requirements:
	@echo "$(BLUE)📋 Generating requirements files...$(RESET)"
	@$(HATCH) run pip freeze > requirements.txt
	@$(HATCH) run pip freeze --exclude-editable > requirements-prod.txt
	@echo "$(GREEN)✅ Requirements files generated!$(RESET)"

# =============================================================================
# Help and Documentation
# =============================================================================

.PHONY: help

# Main help
help:
	@echo "$(BLUE)Cogent AI Agent System - Makefile$(RESET)"
	@echo "$(YELLOW)=====================================$(RESET)"
	@echo ""
	@echo "$(GREEN)Development Commands:$(RESET)"
	@echo "  $(YELLOW)make dev$(RESET)         - Start development server with auto-reload"
	@echo "  $(YELLOW)make dev-debug$(RESET)   - Start development server with debug logging"
	@echo "  $(YELLOW)make install$(RESET)     - Install dependencies"
	@echo "  $(YELLOW)make install-dev$(RESET) - Install development dependencies"
	@echo "  $(YELLOW)make shell$(RESET)       - Activate development shell"
	@echo ""
	@echo "$(GREEN)Code Quality Commands:$(RESET)"
	@echo "  $(YELLOW)make format$(RESET)      - Format code (black, isort, autoflake)"
	@echo "  $(YELLOW)make format-check$(RESET) - Check code formatting"
	@echo "  $(YELLOW)make lint$(RESET)        - Run linters"
	@echo "  $(YELLOW)make lint-fix$(RESET)    - Auto-fix linting issues"
	@echo "  $(YELLOW)make quality$(RESET)     - Run all quality checks"
	@echo ""
	@echo "$(GREEN)Testing Commands:$(RESET)"
	@echo "  $(YELLOW)make test$(RESET)        - Run all tests"
	@echo "  $(YELLOW)make test-unit$(RESET)   - Run unit tests only"
	@echo "  $(YELLOW)make test-integration$(RESET) - Run integration tests only"
	@echo "  $(YELLOW)make test-coverage$(RESET) - Run tests with coverage"
	@echo "  $(YELLOW)make test-watch$(RESET)  - Run tests in watch mode"
	@echo ""
	@echo "$(GREEN)Build Commands:$(RESET)"
	@echo "  $(YELLOW)make build$(RESET)       - Build package"
	@echo "  $(YELLOW)make package$(RESET)     - Build and package for distribution"
	@echo ""
	@echo "$(GREEN)Docker Commands:$(RESET)"
	@echo "  $(YELLOW)make docker-build$(RESET) - Build Docker image"
	@echo "  $(YELLOW)make docker-run$(RESET)   - Run Docker container"
	@echo "  $(YELLOW)make docker-stop$(RESET)  - Stop Docker container"
	@echo ""
	@echo "$(GREEN)Utility Commands:$(RESET)"
	@echo "  $(YELLOW)make clean$(RESET)       - Clean Python cache and build artifacts"
	@echo "  $(YELLOW)make clean-all$(RESET)   - Clean everything including Docker"
	@echo "  $(YELLOW)make check-env$(RESET)   - Check environment setup"
	@echo "  $(YELLOW)make requirements$(RESET) - Generate requirements files"

# Default target
.DEFAULT_GOAL := help 