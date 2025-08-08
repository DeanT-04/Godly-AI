#!/usr/bin/env python3
"""
Installation and setup script for Godly AI system.

This script handles the initial setup and configuration of the development environment.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(command: List[str], cwd: Optional[Path] = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ {' '.join(command)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {' '.join(command)}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is 3.11 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)")
        return False


def check_poetry_installed() -> bool:
    """Check if Poetry is installed."""
    try:
        result = subprocess.run(
            ["poetry", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poetry not found")
        return False


def install_dependencies() -> bool:
    """Install project dependencies using Poetry."""
    print("\n📦 Installing dependencies...")
    return run_command(["poetry", "install"])


def setup_pre_commit() -> bool:
    """Set up pre-commit hooks."""
    print("\n🔧 Setting up pre-commit hooks...")
    return run_command(["poetry", "run", "pre-commit", "install"])


def create_directories() -> bool:
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs",
        "data/raw",
        "data/processed", 
        "data/models",
        "data/experiments"
    ]
    
    project_root = Path(__file__).parent.parent.parent
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")
    
    return True


def run_initial_tests() -> bool:
    """Run initial tests to verify installation."""
    print("\n🧪 Running initial tests...")
    return run_command(["poetry", "run", "pytest", "tests/", "-v", "--tb=short"])


def main():
    """Main setup function."""
    print("🚀 Godly AI System Setup")
    print("=" * 50)
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    
    if not check_python_version():
        print("\n❌ Setup failed: Python 3.11+ required")
        sys.exit(1)
    
    if not check_poetry_installed():
        print("\n❌ Setup failed: Poetry not installed")
        print("Install Poetry: https://python-poetry.org/docs/#installation")
        sys.exit(1)
    
    # Setup steps
    setup_steps = [
        ("Installing dependencies", install_dependencies),
        ("Setting up pre-commit hooks", setup_pre_commit),
        ("Creating directories", create_directories),
        ("Running initial tests", run_initial_tests),
    ]
    
    failed_steps = []
    
    for step_name, step_function in setup_steps:
        print(f"\n{step_name}...")
        if not step_function():
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "=" * 50)
    if failed_steps:
        print("❌ Setup completed with errors:")
        for step in failed_steps:
            print(f"  - {step}")
        sys.exit(1)
    else:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate the virtual environment: poetry shell")
        print("2. Start development: see README.md for usage")
        print("3. Run tests: pytest tests/")


if __name__ == "__main__":
    main()