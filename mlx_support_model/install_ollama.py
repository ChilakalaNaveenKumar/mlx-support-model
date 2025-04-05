#!/usr/bin/env python3
"""
Installation script for setting up Ollama support.
Checks for dependencies and guides users through setup.
"""

import os
import sys
import subprocess
import platform
import logging
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("install")

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or (major == 3 and minor < 8):
        logger.error(f"Python 3.8+ required, found {major}.{minor}")
        return False
    logger.info(f"✓ Python version OK: {major}.{minor}")
    return True

def check_os_compatibility() -> bool:
    """Check if OS is compatible with Ollama."""
    system = platform.system()
    if system not in ["Darwin", "Linux", "Windows"]:
        logger.error(f"Unsupported OS: {system}")
        return False
        
    if system == "Darwin":
        # Check for Apple Silicon
        machine = platform.machine()
        if machine == "arm64":
            logger.info("✓ macOS on Apple Silicon detected")
        else:
            logger.warning(f"⚠ Running on macOS {machine} - Ollama works best on Apple Silicon")
    elif system == "Windows":
        logger.warning("⚠ Windows support for Ollama requires WSL2")
    else:
        logger.info(f"✓ {system} detected")
    
    return True

def run_command(command: List[str]) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            command, 
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def check_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    success, output = run_command(["ollama", "--version"])
    if success:
        logger.info(f"✓ Ollama is installed: {output.strip()}")
        return True
    logger.warning("✗ Ollama is not installed")
    return False

def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    import socket
    try:
        # Try to connect to Ollama's default port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        
        if result == 0:
            logger.info("✓ Ollama server is running")
            return True
        logger.warning("✗ Ollama server is not running")
        return False
    except Exception as e:
        logger.warning(f"✗ Error checking Ollama server: {e}")
        return False

def check_mlx_installed() -> bool:
    """Check if MLX is installed."""
    success = False
    try:
        import mlx
        success = True
        logger.info(f"✓ MLX is installed: {mlx.__version__ if hasattr(mlx, '__version__') else 'unknown version'}")
    except ImportError:
        logger.warning("✗ MLX is not installed")
    
    try:
        import mlx_lm
        success = True
        logger.info(f"✓ MLX-LM is installed: {mlx_lm.__version__ if hasattr(mlx_lm, '__version__') else 'unknown version'}")
    except ImportError:
        logger.warning("✗ MLX-LM is not installed")
    
    return success

def install_ollama() -> bool:
    """Install Ollama based on the operating system."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        logger.info("Installing Ollama for macOS...")
        logger.info("1. Download the latest version from https://ollama.com/download")
        logger.info("2. Open the downloaded file and follow installation instructions")
        
        # Ask if user wants to open the download page
        response = input("Would you like to open the download page? (y/n): ")
        if response.lower() in ['y', 'yes']:
            run_command(["open", "https://ollama.com/download"])
        
        return False  # Manual installation required
        
    elif system == "Linux":
        logger.info("Installing Ollama for Linux...")
        
        # Ask for confirmation
        response = input("Install Ollama using curl | sh? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Installation cancelled")
            return False
            
        # Run the official installation script
        success, output = run_command(["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"])
        logger.info(output)
        return success
        
    elif system == "Windows":
        logger.info("Installing Ollama for Windows...")
        logger.info("Windows requires WSL2 for Ollama:")
        logger.info("1. Enable WSL2: https://docs.microsoft.com/en-us/windows/wsl/install")
        logger.info("2. Install a Linux distribution from Microsoft Store")
        logger.info("3. Install Ollama inside WSL2 with: curl -fsSL https://ollama.ai/install.sh | sh")
        
        # Ask if user wants to open the WSL docs
        response = input("Would you like to open the WSL installation docs? (y/n): ")
        if response.lower() in ['y', 'yes']:
            run_command(["start", "https://docs.microsoft.com/en-us/windows/wsl/install"])
        
        return False  # Manual installation required
        
    else:
        logger.error(f"Unsupported OS: {system}")
        return False

def install_mlx() -> bool:
    """Install MLX and MLX-LM."""
    logger.info("Installing MLX and MLX-LM...")
    
    # Check if pip is available
    success, _ = run_command([sys.executable, "-m", "pip", "--version"])
    if not success:
        logger.error("pip is not available, cannot install packages")
        return False
    
    # Install mlx
    logger.info("Installing mlx...")
    success, output = run_command([sys.executable, "-m", "pip", "install", "-U", "mlx"])
    logger.info(output)
    if not success:
        logger.error("Failed to install mlx")
        return False
    
    # Install mlx-lm
    logger.info("Installing mlx-lm...")
    success, output = run_command([sys.executable, "-m", "pip", "install", "-U", "mlx-lm"])
    logger.info(output)
    if not success:
        logger.error("Failed to install mlx-lm")
        return False
    
    return True

def install_project_requirements() -> bool:
    """Install project-specific requirements."""
    logger.info("Installing project requirements...")
    
    # Check if setup.py exists
    if os.path.exists("setup.py"):
        logger.info("Installing via setup.py in development mode...")
        success, output = run_command([sys.executable, "-m", "pip", "install", "-e", "."])
        logger.info(output)
        return success
    
    # Check if requirements.txt exists
    elif os.path.exists("requirements.txt"):
        logger.info("Installing via requirements.txt...")
        success, output = run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info(output)
        return success
    
    else:
        logger.warning("No setup.py or requirements.txt found")
        return False

def setup_configuration() -> None:
    """Setup or update configuration for using Ollama."""
    logger.info("Setting up configuration...")
    
    # Check if config.py exists
    config_path = os.path.join("mlx_support_model", "config.py")
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return
    
    # Read current configuration
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Check if LLM_PROVIDER is already defined
    if "LLM_PROVIDER =" in config_content:
        # Update provider if needed
        response = input("Set Ollama as the default provider? (y/n): ")
        if response.lower() in ['y', 'yes']:
            # Very simple update approach - for complex changes, would use AST
            lines = config_content.split('\n')
            updated_lines = []
            for line in lines:
                if line.strip().startswith("LLM_PROVIDER ="):
                    updated_lines.append('LLM_PROVIDER = "ollama"')
                else:
                    updated_lines.append(line)
            
            # Write updated config
            with open(config_path, 'w') as f:
                f.write('\n'.join(updated_lines))
            
            logger.info("✓ Configuration updated to use Ollama by default")
    else:
        logger.warning("LLM_PROVIDER setting not found in the configuration file")
        logger.info("Please manually update the configuration to add Ollama support")

def list_available_models() -> None:
    """List available models from Ollama."""
    logger.info("Checking available Ollama models...")
    
    # First check if Ollama is running
    if not check_ollama_running():
        logger.warning("Cannot list models because Ollama server is not running")
        return
    
    # Try to get models using API
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'models' in data and data['models']:
                logger.info("Available Ollama models:")
                for model in data['models']:
                    logger.info(f"  - {model['name']} ({model['size']})")
            else:
                logger.info("No models found. You can pull models with 'ollama pull <model>'")
        else:
            logger.warning(f"Error getting models: {response.status_code}")
    except Exception as e:
        logger.warning(f"Error connecting to Ollama API: {e}")
        logger.info("You can list models with the command: ollama list")

def pull_recommended_models() -> None:
    """Pull recommended models for Ollama."""
    # Check if Ollama is running
    if not check_ollama_running():
        logger.warning("Cannot pull models because Ollama server is not running")
        return
    
    # List of recommended models
    recommended_models = ["llama3", "codellama", "phi3"]
    
    # Ask which models to pull
    logger.info("Recommended models:")
    for i, model in enumerate(recommended_models):
        logger.info(f"  {i+1}. {model}")
    
    response = input("Enter the numbers of models to pull (comma-separated) or 'all': ")
    if not response:
        return
    
    if response.lower() == 'all':
        models_to_pull = recommended_models
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in response.split(',')]
            models_to_pull = [recommended_models[idx] for idx in indices if 0 <= idx < len(recommended_models)]
        except (ValueError, IndexError):
            logger.warning("Invalid input, no models will be pulled")
            return
    
    # Pull each selected model
    for model in models_to_pull:
        logger.info(f"Pulling model: {model}...")
        success, output = run_command(["ollama", "pull", model])
        if success:
            logger.info(f"✓ Successfully pulled {model}")
        else:
            logger.warning(f"✗ Failed to pull {model}: {output}")

def main() -> int:
    """Main function to run the installation script."""
    logger.info("=" * 60)
    logger.info("MLX/Ollama Support Installation")
    logger.info("=" * 60)
    
    # System checks
    if not check_python_version() or not check_os_compatibility():
        logger.error("System requirements not met. Exiting.")
        return 1
    
    # Check for Ollama
    ollama_installed = check_ollama_installed()
    ollama_running = check_ollama_running() if ollama_installed else False
    
    # Check for MLX
    mlx_installed = check_mlx_installed()
    
    # Installation steps
    if not ollama_installed:
        logger.info("\n" + "=" * 60)
        logger.info("Installing Ollama")
        logger.info("=" * 60)
        
        if install_ollama():
            logger.info("✓ Ollama installed successfully")
            ollama_installed = True
        else:
            logger.info("Follow the manual installation instructions")
    
    if not mlx_installed:
        logger.info("\n" + "=" * 60)
        logger.info("Installing MLX")
        logger.info("=" * 60)
        
        if install_mlx():
            logger.info("✓ MLX installed successfully")
            mlx_installed = True
        else:
            logger.warning("Could not install MLX automatically")
    
    # Install project requirements
    logger.info("\n" + "=" * 60)
    logger.info("Project Setup")
    logger.info("=" * 60)
    
    install_project_requirements()
    setup_configuration()
    
    # Model management (if Ollama is running)
    if ollama_installed and not ollama_running:
        logger.info("\nOllama is installed but not running")
        logger.info("Start Ollama before continuing with model management")
    elif ollama_installed and ollama_running:
        logger.info("\n" + "=" * 60)
        logger.info("Ollama Model Management")
        logger.info("=" * 60)
        
        list_available_models()
        
        # Ask if user wants to pull recommended models
        response = input("\nWould you like to pull recommended models now? (y/n): ")
        if response.lower() in ['y', 'yes']:
            pull_recommended_models()
    
    # Final instructions
    logger.info("\n" + "=" * 60)
    logger.info("Installation Complete")
    logger.info("=" * 60)
    
    if not ollama_installed or not ollama_running:
        logger.info("\nTo use Ollama:")
        logger.info("1. Complete the Ollama installation if needed")
        logger.info("2. Start the Ollama server")
        logger.info("3. Pull models with: ollama pull <model_name>")
    
    logger.info("\nTo run the application with Ollama:")
    logger.info("python main.py --provider ollama --model llama3")
    
    if mlx_installed:
        logger.info("\nTo run with MLX:")
        logger.info("python main.py --provider mlx")
    
    logger.info("\nFor more options:")
    logger.info("python main.py --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())