"""
Test project structure and basic imports.

This test ensures that the project structure is correctly set up
and that basic imports work as expected.
"""

import sys
from pathlib import Path

import pytest


class TestProjectStructure:
    """Test the basic project structure and imports."""
    
    def test_project_root_exists(self):
        """Test that project root directory exists."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / "src").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "config").exists()
    
    def test_source_directories_exist(self):
        """Test that all source directories exist."""
        project_root = Path(__file__).parent.parent
        src_dir = project_root / "src"
        
        expected_dirs = [
            "core",
            "memory", 
            "agents",
            "training",
            "interface"
        ]
        
        for directory in expected_dirs:
            assert (src_dir / directory).exists(), f"Directory {directory} not found"
            assert (src_dir / directory / "__init__.py").exists(), f"__init__.py missing in {directory}"
    
    def test_core_subdirectories_exist(self):
        """Test that core subdirectories exist."""
        project_root = Path(__file__).parent.parent
        core_dir = project_root / "src" / "core"
        
        expected_subdirs = [
            "neurons",
            "synapses", 
            "plasticity",
            "topology"
        ]
        
        for subdir in expected_subdirs:
            assert (core_dir / subdir).exists(), f"Core subdirectory {subdir} not found"
            assert (core_dir / subdir / "__init__.py").exists(), f"__init__.py missing in core/{subdir}"
    
    def test_memory_subdirectories_exist(self):
        """Test that memory subdirectories exist."""
        project_root = Path(__file__).parent.parent
        memory_dir = project_root / "src" / "memory"
        
        expected_subdirs = [
            "working",
            "episodic",
            "semantic", 
            "meta"
        ]
        
        for subdir in expected_subdirs:
            assert (memory_dir / subdir).exists(), f"Memory subdirectory {subdir} not found"
            assert (memory_dir / subdir / "__init__.py").exists(), f"__init__.py missing in memory/{subdir}"
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        project_root = Path(__file__).parent.parent
        
        config_files = [
            "pyproject.toml",
            ".pre-commit-config.yaml",
            ".gitignore",
            "config/default.yaml"
        ]
        
        for config_file in config_files:
            assert (project_root / config_file).exists(), f"Config file {config_file} not found"
    
    def test_basic_imports(self):
        """Test that basic imports work."""
        # Add src to path for testing
        project_root = Path(__file__).parent.parent
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Test basic imports
        try:
            import src
            import src.core
            import src.memory
            import src.agents
            import src.training
            import src.interface
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_python_version(self):
        """Test that Python version is 3.11+."""
        assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


class TestDocumentation:
    """Test documentation files."""
    
    def test_readme_exists(self):
        """Test that README.md exists and has content."""
        project_root = Path(__file__).parent.parent
        readme_path = project_root / "README.md"
        
        assert readme_path.exists(), "README.md not found"
        
        content = readme_path.read_text(encoding='utf-8')
        assert len(content) > 100, "README.md appears to be empty or too short"
        assert "Godly AI" in content, "README.md doesn't mention Godly AI"
    
    def test_contributing_exists(self):
        """Test that CONTRIBUTING.md exists."""
        project_root = Path(__file__).parent.parent
        contributing_path = project_root / "CONTRIBUTING.md"
        
        assert contributing_path.exists(), "CONTRIBUTING.md not found"
        
        content = contributing_path.read_text(encoding='utf-8')
        assert len(content) > 100, "CONTRIBUTING.md appears to be empty or too short"
    
    def test_license_exists(self):
        """Test that LICENSE file exists."""
        project_root = Path(__file__).parent.parent
        license_path = project_root / "LICENSE"
        
        assert license_path.exists(), "LICENSE file not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])