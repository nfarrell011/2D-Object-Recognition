# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.28.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.28.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/build

# Include any dependencies generated for this target.
include CMakeFiles/object_rec.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/object_rec.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/object_rec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/object_rec.dir/flags.make

CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.o: CMakeFiles/object_rec.dir/flags.make
CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.o: /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_util/csv_util.cpp
CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.o: CMakeFiles/object_rec.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.o -MF CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.o.d -o CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.o -c /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_util/csv_util.cpp

CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_util/csv_util.cpp > CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.i

CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/csv_util/csv_util.cpp -o CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.s

CMakeFiles/object_rec.dir/utils.cpp.o: CMakeFiles/object_rec.dir/flags.make
CMakeFiles/object_rec.dir/utils.cpp.o: /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/utils.cpp
CMakeFiles/object_rec.dir/utils.cpp.o: CMakeFiles/object_rec.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/object_rec.dir/utils.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/object_rec.dir/utils.cpp.o -MF CMakeFiles/object_rec.dir/utils.cpp.o.d -o CMakeFiles/object_rec.dir/utils.cpp.o -c /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/utils.cpp

CMakeFiles/object_rec.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/object_rec.dir/utils.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/utils.cpp > CMakeFiles/object_rec.dir/utils.cpp.i

CMakeFiles/object_rec.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/object_rec.dir/utils.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/utils.cpp -o CMakeFiles/object_rec.dir/utils.cpp.s

CMakeFiles/object_rec.dir/object_rec.cpp.o: CMakeFiles/object_rec.dir/flags.make
CMakeFiles/object_rec.dir/object_rec.cpp.o: /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/object_rec.cpp
CMakeFiles/object_rec.dir/object_rec.cpp.o: CMakeFiles/object_rec.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/object_rec.dir/object_rec.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/object_rec.dir/object_rec.cpp.o -MF CMakeFiles/object_rec.dir/object_rec.cpp.o.d -o CMakeFiles/object_rec.dir/object_rec.cpp.o -c /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/object_rec.cpp

CMakeFiles/object_rec.dir/object_rec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/object_rec.dir/object_rec.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/object_rec.cpp > CMakeFiles/object_rec.dir/object_rec.cpp.i

CMakeFiles/object_rec.dir/object_rec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/object_rec.dir/object_rec.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/object_rec.cpp -o CMakeFiles/object_rec.dir/object_rec.cpp.s

# Object files for target object_rec
object_rec_OBJECTS = \
"CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.o" \
"CMakeFiles/object_rec.dir/utils.cpp.o" \
"CMakeFiles/object_rec.dir/object_rec.cpp.o"

# External object files for target object_rec
object_rec_EXTERNAL_OBJECTS =

object_rec: CMakeFiles/object_rec.dir/csv_util/csv_util.cpp.o
object_rec: CMakeFiles/object_rec.dir/utils.cpp.o
object_rec: CMakeFiles/object_rec.dir/object_rec.cpp.o
object_rec: CMakeFiles/object_rec.dir/build.make
object_rec: /opt/homebrew/lib/libopencv_gapi.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_stitching.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_alphamat.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_aruco.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_bgsegm.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_bioinspired.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_ccalib.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_dnn_objdetect.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_dnn_superres.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_dpm.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_face.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_freetype.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_fuzzy.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_hfs.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_img_hash.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_intensity_transform.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_line_descriptor.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_mcc.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_quality.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_rapid.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_reg.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_rgbd.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_saliency.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_sfm.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_stereo.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_structured_light.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_superres.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_surface_matching.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_tracking.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_videostab.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_viz.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_wechat_qrcode.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_xfeatures2d.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_xobjdetect.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_xphoto.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_shape.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_highgui.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_datasets.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_plot.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_text.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_ml.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_phase_unwrapping.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_optflow.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_ximgproc.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_video.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_videoio.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_imgcodecs.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_objdetect.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_calib3d.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_dnn.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_features2d.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_flann.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_photo.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_imgproc.4.9.0.dylib
object_rec: /opt/homebrew/lib/libopencv_core.4.9.0.dylib
object_rec: CMakeFiles/object_rec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable object_rec"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/object_rec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/object_rec.dir/build: object_rec
.PHONY : CMakeFiles/object_rec.dir/build

CMakeFiles/object_rec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/object_rec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/object_rec.dir/clean

CMakeFiles/object_rec.dir/depend:
	cd /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3 /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3 /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/build /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/build /Users/nelsonfarrell/Documents/Northeastern/5330/problem_sets/problem_set_3/build/CMakeFiles/object_rec.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/object_rec.dir/depend
