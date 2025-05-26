"""Set up script to perform AOT compilation when building the package."""
from setuptools import setup
from setuptools.command.build_py import build_py
import subprocess
import os
import sys

class BuildPyWithNumba(build_py):
    def run(self):
        # First run the normal build_py
        super().run()
        
        # Then compile the Numba AOT module
        script_path = os.path.join("src", "pymatmatmul", "numba_compiled_matmul.py")
        print(f"Compiling Numba AOT module at: {script_path}")
        
        if not os.path.exists(script_path):
            print(f"ERROR: Numba script not found at {script_path}")
            sys.exit(1)
            
        try:
            result = subprocess.run([sys.executable, script_path], 
                                  check=True, 
                                  capture_output=True, 
                                  text=True,
                                  cwd=os.getcwd())
            print("Numba AOT compilation successful!")
            if result.stdout:
                print("STDOUT:", result.stdout)
                
            # Copy the compiled shared library to the build directory
            import glob
            import shutil
            
            # Look for compiled shared libraries
            so_files = glob.glob("src/pymatmatmul/*.so") + glob.glob("src/pymatmatmul/*.dll") + glob.glob("src/pymatmatmul/*.dylib")
            
            if so_files:
                build_lib_dir = os.path.join(self.build_lib, "pymatmatmul")
                os.makedirs(build_lib_dir, exist_ok=True)
                
                for so_file in so_files:
                    dest = os.path.join(build_lib_dir, os.path.basename(so_file))
                    shutil.copy2(so_file, dest)
                    print(f"Copied {so_file} to {dest}")
            else:
                print("Warning: No compiled shared libraries found")
                
        except subprocess.CalledProcessError as e:
            print(f"Numba AOT compilation failed: {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            raise

setup(
    cmdclass={"build_py": BuildPyWithNumba}
)
