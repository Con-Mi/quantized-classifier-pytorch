# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/s27909/clion-2019.2.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/s27909/clion-2019.2.4/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/s27909/projects/deep_learning_prjs/_pytorch/_quantized_classifier/quantized-classifier-pytorch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/s27909/projects/deep_learning_prjs/_pytorch/_quantized_classifier/quantized-classifier-pytorch/cmake-build-debug

# Utility rule file for NightlyCoverage.

# Include the progress variables for this target.
include CMakeFiles/NightlyCoverage.dir/progress.make

CMakeFiles/NightlyCoverage:
	/home/s27909/clion-2019.2.4/bin/cmake/linux/bin/ctest -D NightlyCoverage

NightlyCoverage: CMakeFiles/NightlyCoverage
NightlyCoverage: CMakeFiles/NightlyCoverage.dir/build.make

.PHONY : NightlyCoverage

# Rule to build all files generated by this target.
CMakeFiles/NightlyCoverage.dir/build: NightlyCoverage

.PHONY : CMakeFiles/NightlyCoverage.dir/build

CMakeFiles/NightlyCoverage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NightlyCoverage.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NightlyCoverage.dir/clean

CMakeFiles/NightlyCoverage.dir/depend:
	cd /home/s27909/projects/deep_learning_prjs/_pytorch/_quantized_classifier/quantized-classifier-pytorch/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/s27909/projects/deep_learning_prjs/_pytorch/_quantized_classifier/quantized-classifier-pytorch /home/s27909/projects/deep_learning_prjs/_pytorch/_quantized_classifier/quantized-classifier-pytorch /home/s27909/projects/deep_learning_prjs/_pytorch/_quantized_classifier/quantized-classifier-pytorch/cmake-build-debug /home/s27909/projects/deep_learning_prjs/_pytorch/_quantized_classifier/quantized-classifier-pytorch/cmake-build-debug /home/s27909/projects/deep_learning_prjs/_pytorch/_quantized_classifier/quantized-classifier-pytorch/cmake-build-debug/CMakeFiles/NightlyCoverage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/NightlyCoverage.dir/depend

