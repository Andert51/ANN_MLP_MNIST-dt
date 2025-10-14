"""
Verificar instalación y configuración del proyecto
"""
import sys
from pathlib import Path

# Colors for terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_colored(message, color):
    print(f"{color}{message}{Colors.ENDC}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print_colored("\n=== Checking Python Version ===", Colors.HEADER)
    
    if version.major >= 3 and version.minor >= 8:
        print_colored(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)", Colors.OKGREEN)
        return True
    else:
        print_colored(f"✗ Python {version.major}.{version.minor}.{version.micro} (Requires 3.8+)", Colors.FAIL)
        return False


def check_dependencies():
    """Check required packages"""
    print_colored("\n=== Checking Dependencies ===", Colors.HEADER)
    
    required = {
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'rich': 'rich',
        'pandas': 'pandas',
        'scipy': 'scipy'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print_colored(f"✓ {package}", Colors.OKGREEN)
        except ImportError:
            print_colored(f"✗ {package} (Missing)", Colors.FAIL)
            missing.append(package)
    
    if missing:
        print_colored(f"\nInstall missing packages:", Colors.WARNING)
        print_colored(f"pip install {' '.join(missing)}", Colors.OKCYAN)
        return False
    
    return True


def check_project_structure():
    """Check project structure"""
    print_colored("\n=== Checking Project Structure ===", Colors.HEADER)
    
    required_dirs = [
        'src',
        'scripts',
        'output',
        'output/images',
        'output/data',
        'docs',
        'test'
    ]
    
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/data_loader.py',
        'src/mlp_model.py',
        'src/experiments.py',
        'src/visualizations.py',
        'src/reports.py',
        'src/ui.py',
        'main.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_ok = True
    
    # Check directories
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print_colored(f"✓ {dir_path}/", Colors.OKGREEN)
        else:
            print_colored(f"✗ {dir_path}/ (Missing)", Colors.FAIL)
            all_ok = False
    
    # Check files
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print_colored(f"✓ {file_path}", Colors.OKGREEN)
        else:
            print_colored(f"✗ {file_path} (Missing)", Colors.FAIL)
            all_ok = False
    
    return all_ok


def check_imports():
    """Check if project modules can be imported"""
    print_colored("\n=== Checking Project Imports ===", Colors.HEADER)
    
    modules = [
        'src.config',
        'src.data_loader',
        'src.mlp_model',
        'src.experiments',
        'src.visualizations',
        'src.reports',
        'src.ui'
    ]
    
    all_ok = True
    
    for module in modules:
        try:
            __import__(module)
            print_colored(f"✓ {module}", Colors.OKGREEN)
        except Exception as e:
            print_colored(f"✗ {module}: {str(e)}", Colors.FAIL)
            all_ok = False
    
    return all_ok


def test_basic_functionality():
    """Test basic functionality"""
    print_colored("\n=== Testing Basic Functionality ===", Colors.HEADER)
    
    try:
        # Test config
        from src.config import MLPConfig, DatasetConfig
        config = MLPConfig(hidden_layers=[64, 32])
        print_colored("✓ Configuration classes", Colors.OKGREEN)
        
        # Test activation functions
        from src.mlp_model import ActivationFunction
        import numpy as np
        x = np.array([1, 2, 3])
        _ = ActivationFunction.sigmoid(x)
        print_colored("✓ Activation functions", Colors.OKGREEN)
        
        # Test noise generator
        from src.data_loader import NoiseGenerator
        X = np.random.rand(10, 784)
        _ = NoiseGenerator.add_gaussian_noise(X, noise_level=0.1)
        print_colored("✓ Noise generator", Colors.OKGREEN)
        
        # Test MLP initialization
        from src.mlp_model import MLPClassifier
        model = MLPClassifier(config)
        print_colored("✓ MLP initialization", Colors.OKGREEN)
        
        return True
        
    except Exception as e:
        print_colored(f"✗ Error: {str(e)}", Colors.FAIL)
        return False


def print_summary(checks):
    """Print summary"""
    print_colored("\n" + "="*60, Colors.HEADER)
    print_colored("VERIFICATION SUMMARY", Colors.HEADER)
    print_colored("="*60, Colors.HEADER)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        color = Colors.OKGREEN if passed else Colors.FAIL
        print_colored(f"{status} - {check_name}", color)
    
    print_colored("\n" + "="*60, Colors.HEADER)
    
    if all_passed:
        print_colored("\n🎉 All checks passed! Ready to start experimenting.", Colors.OKGREEN)
        print_colored("\nQuick start:", Colors.OKCYAN)
        print_colored("  python main.py", Colors.BOLD)
        print_colored("  python scripts/quick_start.py", Colors.BOLD)
        print_colored("\nFor help, see:", Colors.OKCYAN)
        print_colored("  README.md", Colors.BOLD)
        print_colored("  docs/QUICKSTART.md", Colors.BOLD)
    else:
        print_colored("\n⚠️  Some checks failed. Please fix the issues above.", Colors.WARNING)
        print_colored("\nTo install dependencies:", Colors.OKCYAN)
        print_colored("  pip install -r requirements.txt", Colors.BOLD)


def main():
    """Main verification"""
    print_colored("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║         MLP-MNIST Project Verification Tool              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """, Colors.OKCYAN)
    
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Project Structure": check_project_structure(),
        "Module Imports": check_imports(),
        "Basic Functionality": test_basic_functionality()
    }
    
    print_summary(checks)


if __name__ == "__main__":
    main()
