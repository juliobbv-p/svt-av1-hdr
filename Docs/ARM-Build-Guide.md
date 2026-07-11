[Top level](../README.md)

# Build and Install for AArch64 targets

## Builds using cross compilers

- __Build Requirements__
  - CMake 3.23 or later
  - Toolchain (Clang or [GCC](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads))
  - For best performance the latest version of a given compiler should be used.

### GCC builds

- __Build Instructions__
  - Run the following commands
    - Release Builds

      ```bash
      cmake -B Bin/Release -S . -DCMAKE_BUILD_TYPE="Release" -DCMAKE_TOOLCHAIN_FILE="./cmake/toolchains/aarch64_toolchain.cmake"
      cmake --build Bin/Release -j
      ```

    - Debug Builds

      ```bash
      cmake -B Bin/Debug -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_TOOLCHAIN_FILE="./cmake/toolchains/aarch64_toolchain.cmake"
      cmake --build Bin/Debug -j
      ```

### Clang builds

- Create a new file called `aarch64_clang_toolchain.cmake` and put the following in it (change the clang path to your preferred version):
  ```bash
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR arm64)
  set(TRIPLE aarch64-linux-gnu)
  set(CMAKE_C_COMPILER clang-19)
  set(CMAKE_C_COMPILER_TARGET ${TRIPLE})
  set(CMAKE_CXX_COMPILER clang++-19)
  set(CMAKE_CXX_COMPILER_TARGET ${TRIPLE})
  set(CMAKE_ASM_FLAGS "--target=${TRIPLE}")
  ```
- Then Release/Debug builds commands are the same as for GCC.

## Native builds on an AArch64 Linux machine

- __Build Requirements__
  - GCC 8.1.0 or later
  - CMake 3.23 or later

- __Build Instructions__
  - Run the following commands
    - Release Builds

      ```bash
      cmake -B Bin/Release -S . -DCMAKE_BUILD_TYPE="Release"
      cmake --build Bin/Release -j
      ```

    - Debug Builds

      ```bash
      cmake -B Bin/Debug -S . -DCMAKE_BUILD_TYPE="Debug"
      cmake --build Bin/Debug -j
      ```

## Binaries Location

- Binaries can be found under `<repo dir>/Bin/Release` or `<repo dir>/Bin/Debug`, depending on whether Debug or Release were selected in the build mode.

## Installation

  For the binaries to operate properly on your system, the following conditions have to be met:

- On any of the Linux* Operating Systems listed above, copy the binaries under a location of your choice.
- Change the permissions on the sample application `SvtAV1EncApp` executable by running the command: `chmod +x SvtAv1EncApp`
- cd into your chosen location
- Run the sample application to encode

   ```bash
   ./SvtAv1EncApp -i [in.yuv] -w [width] -h [height] -b [out.ivf]
   ```

- Sample application supports reading from pipe. For example

   ```bash
   ffmpeg -i [input.mp4] -nostdin -f rawvideo -pix_fmt yuv420p - | ./SvtAv1EncApp -i stdin -n [number_of_frames_to_encode] -w [width] -h [height]
   ```
