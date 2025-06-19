# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import pathlib
import shlex
import shutil
import re
import subprocess
from distutils.file_util import copy_file

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig


# Retrieve the version from Git tags
def get_version():
    if 'CUSTOM_KSANA_VERSION' in os.environ:
        custom_ksana_version = os.environ['CUSTOM_KSANA_VERSION']
        return custom_ksana_version
    git_describe_output = subprocess.run(
        ["git", "describe", "--tags", "--always"],
        stdout=subprocess.PIPE,
        text=True).stdout.strip()
    match = re.match(r'^(v[\d\.]+(?:[-\.]?(?:rc|post|beta|alpha)[-\.\d]*)?)', git_describe_output)
    print(match)
    if match:
        return match.group(1)
    else:
        return git_describe_output


class CMakeExtension(Extension):
    """
    An extension module that will be built using CMake.
    We override the default behavior since we don't have sources here.
    """
    def __init__(self, name):
        super().__init__(name, sources=[])


def is_run_on_npu_device() -> bool:
    """
    Check if the current environment is running on an NPU device.
    This is done by checking if 'torch_npu' is in the installed packages.
    """
    import pkgutil
    return "torch_npu" in [pkg.name for pkg in pkgutil.iter_modules()]


class BuildExt(build_ext_orig):
    """
    A custom build extension for building CMake-based projects.
    """
    def run(self):
        """
        Run method that builds the CMake extension before proceeding with the standard build.
        """
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        """
        Build the extension using CMake.
        """
        # Absolute path to the current working directory
        cwd = pathlib.Path().absolute()

        # Ensure the build temporary directory exists
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Ensure the extension output directory exists
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # Collect all the implemented endpoints from the 'endpoints/warpper' directory
        endpoints_dir = cwd / "src" / "ksana_llm" / "endpoints" / "wrapper"
        implemented_endpoints = [
            entry.name for entry in endpoints_dir.iterdir() if entry.is_dir()
        ]

        # Configuration: 'Debug' or 'Release'
        config = "Debug" if self.debug else "Release"

        # Parse any additional CMake arguments from the environment variable 'CMAKE_ARGS'
        cmake_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))

        # Initialize a dictionary to hold all the CMake options
        cmake_options = {
            "CMAKE_LIBRARY_OUTPUT_DIRECTORY": str(extdir.parent.absolute()),
            "CMAKE_BUILD_TYPE": config,
            "WITH_TESTING": "OFF",
        }

        # Determine if running on an NPU device, and set options accordingly
        if is_run_on_npu_device():
            # NPU device specific options
            cmake_options.update({
                "WITH_CUDA": "OFF",
                "WITH_ACL": "ON",
            })
        else:
            # CUDA device specific options
            cmake_options.update({
                "WITH_CUDA": "ON",
                "SM": "80,86,89",  # Specify the supported GPU architectures
            })

        # Dictionary to track the status (enabled/disabled) of each endpoint
        endpoint_status = {}

        # Iterate over implemented endpoints to set their build options
        for endpoint in implemented_endpoints:
            # Formulate the option name, e.g., 'WITH_GRPC_ENDPOINT'
            option_name = f"WITH_{endpoint.upper()}_ENDPOINT"
            # Default to enabling the endpoint
            endpoint_status[endpoint] = True
            cmake_options[option_name] = "ON"

            # Check if the endpoint is explicitly turned off via '-D' argument
            turned_off_option = f"-D{option_name}=OFF"
            if turned_off_option in cmake_args:
                # If the option is found in cmake_args, disable the endpoint
                endpoint_status[endpoint] = False
                cmake_options[option_name] = "OFF"

        # Convert the cmake_options dictionary into command-line arguments
        cmake_args.extend([f"-D{key}={value}" for key, value in cmake_options.items()])

        # Build arguments for 'cmake --build'
        build_args = ["--config", config]
        # Use all available CPU cores for parallel build
        build_args += ["--", f"-j{os.cpu_count()}"]

        # Change to the build temporary directory and run CMake configuration
        os.chdir(build_temp)
        self.spawn(["cmake", str(cwd)] + cmake_args)
        if not self.dry_run:
            # Build the project
            self.spawn(["cmake", "--build", "."] + build_args)

        # Return to the original directory
        os.chdir(cwd)

        # After building, copy the necessary files to the package directory
        self.copy_built_libraries(extdir, build_temp, cwd, implemented_endpoints, endpoint_status)
        self.copy_python_files_and_dirs(extdir, cwd, implemented_endpoints, endpoint_status)

    def copy_built_libraries(self, extdir, build_temp, cwd, implemented_endpoints, endpoint_status):
        """
        Copy the built libraries to the appropriate locations in the package.
        """
        # Directories for the built libraries
        deps_lib = cwd / "src" / "ksana_llm" / "python" / "lib"
        build_temp_lib = build_temp / "lib"

        # Ensure the dependencies library directory exists
        deps_lib.mkdir(parents=True, exist_ok=True)

        # List of target libraries to copy
        target_libs = ["libtorch_serving.so", "libloguru.so"]

        # Include endpoint libraries that are enabled
        for endpoint in implemented_endpoints:
            if endpoint_status[endpoint] and endpoint != "triton":
                target_libs.append(f"lib{endpoint}_endpoint.so")

        # Copy libraries to both the local development directory and the package directory
        for target_lib in target_libs:
            src_lib = build_temp_lib / target_lib
            dst_lib_deps = deps_lib / target_lib
            dst_lib_ext = extdir.parent / target_lib

            # Copy to the dependencies library directory for local development
            copy_file(str(src_lib), str(dst_lib_deps))
            # Copy to the package directory for inclusion in the wheel
            copy_file(str(src_lib), str(dst_lib_ext))

    def copy_python_files_and_dirs(self, extdir, cwd, implemented_endpoints, endpoint_status):
        """
        Copy the required Python files and directories into the package.
        """
        # List of directories to copy into the package
        need_dirs = ["weight_map", "ksana_plugin"]

        # Copy each required directory
        for need_dir in need_dirs:
            src_dir = cwd / "src" / "ksana_llm" / "python" / need_dir
            dst_dir = extdir.parent / "ksana_llm" / need_dir
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        # Copy endpoint configuration files if they exist
        for endpoint in implemented_endpoints:
            if endpoint_status[endpoint]:
                if endpoint == "triton":
                    # For 'triton backend', copy the entire endpoint directory
                    src_dir = cwd / "src" / "ksana_llm" / "endpoints" / "wrapper" / endpoint
                    dst_dir = extdir.parent / "ksana_llm" / "triton_backend"
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

                    src_conf = cwd / "examples" /  "ksana_llm.yaml"
                    dst_conf = extdir.parent / "ksana_llm/triton_backend/config/ksana_llm/1/ksana_llm.yaml"
                    shutil.copyfile(src_conf, dst_conf)
                else:
                    # For other endpoints, copy the 'rpc_config' directory
                    src_dir = cwd / "src" / "ksana_llm" / "endpoints" / "wrapper" / endpoint / "rpc_config"
                    dst_dir = extdir.parent / "ksana_llm" / "rpc_config"
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        # List of required Python files to copy into the package
        need_files = [
            "serving_server.py",
            "serving_generate_client.py",
            "serving_forward_client.py",
        ]

        # Copy each required Python file
        for need_file in need_files:
            src_file = cwd / "src" / "ksana_llm" / "python" / need_file
            dst_dir = extdir.parent / "ksana_llm"
            copy_file(str(src_file), str(dst_dir))

        # Copy triton cubin kernels
        if 'KSANA_TRITON_KERNEL_PATH' in os.environ:
            triton_kernel_dir = os.environ['KSANA_TRITON_KERNEL_PATH'] + "/triton_kernel_files"
        else:
            home_dir = os.path.expanduser("~")
            triton_kernel_dir = f"{home_dir}/.cache/KsanaLLM/triton_kernel_files"
        dst_dir = extdir.parent / "ksana_llm" / "triton_kernel_files"
        shutil.copytree(triton_kernel_dir, dst_dir, dirs_exist_ok=True)

# Setup configuration for the ksana_llm package
setup(
    name="ksana_llm",
    version=get_version(),
    author="ksana_llm",
    author_email="ksana_llm@tencent.com",
    description="Ksana LLM inference server",
    platforms="python3",
    url="https://xxx/KsanaLLM/",
    packages=find_packages("src/ksana_llm/python"),
    package_dir={
        "": "src/ksana_llm/python",
    },
    ext_modules=[CMakeExtension("ksana_llm")],
    python_requires=">=3",
    install_requires=open("requirements.txt").readlines(),
    cmdclass={
        "build_ext": BuildExt,
    },
)
