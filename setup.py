import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_namespace_packages


def build_docker():
    """Build Docker container"""
    print("Building Docker image...")
    try:
        subprocess.run(["docker", "build", "-t", "ants-bidsapp", "."], check=True)
        print("Docker image built successfully")
    except subprocess.CalledProcessError as e:
        print(f"Docker build failed: {e}")
        return False
    return True


def build_singularity(output_path=None):
    """Build Singularity/Apptainer container"""
    print("Building container image...")
    try:
        # Check for apptainer first (more common on clusters), then singularity
        if (
            subprocess.run(["which", "apptainer"], capture_output=True).returncode == 0
        ):
            print("\nDetected Apptainer on cluster environment.")
            print("For cluster environments, please build directly with apptainer:")
            print("apptainer build --fakeroot ants-bidsapp.sif Singularity\n")
            return False
        elif (
            subprocess.run(["which", "singularity"], capture_output=True).returncode == 0
        ):
            container_cmd = "singularity"
        else:
            print("Neither apptainer nor singularity found. Cannot build image.")
            return False

        # Use custom output path if provided, otherwise use default
        output_file = output_path if output_path else "ants-bidsapp.sif"
        output_file = str(Path(output_file).resolve())
        
        # Build command
        cmd = [container_cmd, "build"]
        
        # For regular Singularity installations, try fakeroot if available
        if subprocess.run(["which", "fakeroot"], capture_output=True).returncode == 0:
            cmd.append("--fakeroot")
        
        # Add output file and Singularity definition
        cmd.extend([output_file, "Singularity"])
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the build command
        subprocess.run(cmd, check=True)
        print(f"Container image built successfully at: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        print("\nFor cluster environments, please build directly with apptainer:")
        print("apptainer build --remote ants-bidsapp.sif Singularity")
        print("or")
        print("apptainer build --fakeroot ants-bidsapp.sif Singularity")
        return False


def init_git_submodules():
    """Initialize git submodules if .git directory exists and --init-git flag is set"""
    if "--init-git" in sys.argv:
        if Path(".git").exists():
            print("Initializing git submodules...")
            try:
                subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
                print("Git submodules initialized successfully")
            except subprocess.CalledProcessError as e:
                print(f"Git submodule initialization failed: {e}")
                print("Continuing without git submodules...")
        else:
            print("No .git directory found, skipping git submodule initialization")
    else:
        print("Skipping git submodule initialization (use --init-git to enable)")


def print_usage():
    """Print usage information"""
    print("Usage:")
    print("  python setup.py install          - Install the package")
    print("  python setup.py docker          - Build Docker container")
    print("  python setup.py singularity     - Build Singularity/Apptainer container")
    print("  python setup.py containers      - Build both Docker and Singularity containers")
    print("  python setup.py --init-git      - Initialize git submodules")
    print("\nFor more information, run: python setup.py --help")


def install_antspyx():
    """Install ANTsPy using conda or pip"""
    print("Installing ANTsPy...")
    
    # Try conda first
    if subprocess.run(["which", "conda"], capture_output=True).returncode == 0:
        print("Using conda to install ANTsPy...")
        try:
            subprocess.run(["conda", "install", "-y", "-c", "conda-forge", "antspyx"], check=True)
            print("ANTsPy installed successfully with conda")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Conda installation failed: {e}")
            print("Trying pip installation...")
    
    # If conda fails or not available, try pip
    try:
        # Check if pip is available
        if subprocess.run(["which", "pip"], capture_output=True).returncode != 0:
            print("Pip not available in this environment. Skipping ANTsPy installation.")
            print("Please install ANTsPy manually before running the tests:")
            print("1. Using conda: conda install -c conda-forge antspyx")
            print("2. Using pip: pip install antspyx")
            return False
            
        subprocess.run(["pip", "install", "antspyx"], check=True)
        print("ANTsPy installed successfully with pip")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Pip installation failed: {e}")
        print("\nFailed to install ANTsPy. Please install it manually:")
        print("1. Using conda: conda install -c conda-forge antspyx")
        print("2. Using pip: pip install antspyx")
        return False


# Handle dependency conflicts by defining dependencies with proper constraints
install_requires = [
    "nibabel>=5.0.0",
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "pytest>=7.0.0",
]

# Check if we're being called with a container build command
if len(sys.argv) > 1 and sys.argv[1] in ["docker", "singularity", "containers"]:
    command = sys.argv[1]
    # Remove the custom argument so setup() doesn't see it
    sys.argv.pop(1)

    if command == "docker":
        build_docker()
    elif command == "singularity":
        # Check for custom output path in the next argument
        output_path = None
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            output_path = sys.argv.pop(1)
        build_singularity(output_path)
    elif command == "containers":
        build_docker()
        build_singularity()

    # Exit if we were just building containers
    if len(sys.argv) == 1:
        sys.exit(0)
elif len(sys.argv) == 1:
    # If no arguments provided, show usage
    print_usage()
    sys.exit(1)

# Initialize git submodules only if explicitly requested
init_git_submodules()

# Install ANTsPy before proceeding with setup
# Make this optional - don't fail if ANTsPy can't be installed
install_antspyx()

# Print note about deprecated setup.py install
if "install" in sys.argv:
    print("\nNote: setup.py install is deprecated. For a more modern approach, consider using:")
    print("  pip install -e .")
    print("or")
    print("  python -m pip install -e .")

setup(
    name="ants-bidsapp",
    version="0.1.0",
    description="BIDS App for ANTs Segmentation with NIDM Output",
    author="ReproNim",
    author_email="repronim@gmail.com",
    packages=find_namespace_packages(include=["src", "src.*"]),
    include_package_data=True,
    license="MIT",
    url="https://github.com/ReproNim/ants-bidsapp",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "ants-bidsapp=src.run:main",
        ],
    },
    python_requires=">=3.9",
    install_requires=install_requires,
) 