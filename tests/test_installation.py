#!/usr/bin/env python3
"""
Installation verification test for Neural-Decoder.

This test checks that all required dependencies are properly installed
and the project structure is correct.
"""

import sys
import os
import importlib
from pathlib import Path

def test_python_version():
    """Test that Python version is 3.8 or higher."""
    major, minor = sys.version_info[:2]
    assert major == 3 and minor >= 8, f"Python 3.8+ required, got {major}.{minor}"
    print("✓ Python version check passed")

def test_required_packages():
    """Test that all required packages can be imported."""
    required_packages = [
        'numpy',
        'pandas', 
        'scipy',
        'sklearn',
        'matplotlib',
        'torch',
        'tensorflow',
        'brainflow',
        'pyqtgraph',
        'yaml'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {e}")
            return False
    
    return True

def test_project_structure():
    """Test that the project directory structure is correct."""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        'src',
        'src/data_collection',
        'src/signal_processing', 
        'src/models',
        'src/visualization',
        'src/utils',
        'scripts',
        'configs',
        'data',
        'models',
        'docs',
        'examples',
        'tests'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ Directory {dir_path} exists")
        else:
            print(f"✗ Missing directory: {dir_path}")
            return False
    
    return True

def test_config_file():
    """Test that the configuration file exists and can be loaded."""
    try:
        import yaml
        config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
        
        if not config_path.exists():
            print("✗ Configuration file not found")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check for required sections
        required_sections = ['device', 'signal_processing', 'ml']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
                
        print("✓ Configuration file loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False

def test_src_imports():
    """Test that source modules can be imported."""
    # Add src to path
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.insert(0, str(src_path))
    
    try:
        from utils.config_loader import config
        print("✓ Configuration loader imported successfully")
        
        # Test that config can be accessed
        device_config = config.get_device_config()
        assert isinstance(device_config, dict), "Device config should be a dictionary"
        print("✓ Configuration access test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to import source modules: {e}")
        return False

def test_script_files():
    """Test that main script files exist."""
    scripts_dir = Path(__file__).parent.parent / 'scripts'
    
    required_scripts = [
        'train_model.py',
        'realtime_detection.py', 
        'alpha_visualization.py'
    ]
    
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✓ Script {script} exists")
        else:
            print(f"✗ Missing script: {script}")
            return False
            
    return True

def main():
    """Run all installation tests."""
    print("Neural-Decoder Installation Test")
    print("=" * 40)
    
    tests = [
        test_python_version,
        test_required_packages,
        test_project_structure,
        test_config_file,
        test_src_imports,
        test_script_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Installation looks good.")
        print("\nNext steps:")
        print("1. Configure your device MAC address in configs/config.yaml")
        print("2. Run: python scripts/train_model.py")
        return True
    else:
        print("❌ Some tests failed. Please check the installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 