# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.25

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "F:\Programming\CLion 2023.1.2\bin\cmake\win\x64\bin\cmake.exe"

# The command to remove a file.
RM = "F:\Programming\CLion 2023.1.2\bin\cmake\win\x64\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = F:\Programming\Recreational\Makina

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = F:\Programming\Recreational\Makina\bin\debug

# Include any dependencies generated for this target.
include CMakeFiles/Makina.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Makina.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Makina.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Makina.dir/flags.make

CMakeFiles/Makina.dir/src/main.c.obj: CMakeFiles/Makina.dir/flags.make
CMakeFiles/Makina.dir/src/main.c.obj: F:/Programming/Recreational/Makina/src/main.c
CMakeFiles/Makina.dir/src/main.c.obj: CMakeFiles/Makina.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=F:\Programming\Recreational\Makina\bin\debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/Makina.dir/src/main.c.obj"
	"F:\Programming\CLion 2023.1.2\bin\mingw\bin\gcc.exe" $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/Makina.dir/src/main.c.obj -MF CMakeFiles\Makina.dir\src\main.c.obj.d -o CMakeFiles\Makina.dir\src\main.c.obj -c F:\Programming\Recreational\Makina\src\main.c

CMakeFiles/Makina.dir/src/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Makina.dir/src/main.c.i"
	"F:\Programming\CLion 2023.1.2\bin\mingw\bin\gcc.exe" $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E F:\Programming\Recreational\Makina\src\main.c > CMakeFiles\Makina.dir\src\main.c.i

CMakeFiles/Makina.dir/src/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Makina.dir/src/main.c.s"
	"F:\Programming\CLion 2023.1.2\bin\mingw\bin\gcc.exe" $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S F:\Programming\Recreational\Makina\src\main.c -o CMakeFiles\Makina.dir\src\main.c.s

# Object files for target Makina
Makina_OBJECTS = \
"CMakeFiles/Makina.dir/src/main.c.obj"

# External object files for target Makina
Makina_EXTERNAL_OBJECTS =

Makina.exe: CMakeFiles/Makina.dir/src/main.c.obj
Makina.exe: CMakeFiles/Makina.dir/build.make
Makina.exe: CMakeFiles/Makina.dir/linkLibs.rsp
Makina.exe: CMakeFiles/Makina.dir/objects1
Makina.exe: CMakeFiles/Makina.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=F:\Programming\Recreational\Makina\bin\debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable Makina.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Makina.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Makina.dir/build: Makina.exe
.PHONY : CMakeFiles/Makina.dir/build

CMakeFiles/Makina.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Makina.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Makina.dir/clean

CMakeFiles/Makina.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" F:\Programming\Recreational\Makina F:\Programming\Recreational\Makina F:\Programming\Recreational\Makina\bin\debug F:\Programming\Recreational\Makina\bin\debug F:\Programming\Recreational\Makina\bin\debug\CMakeFiles\Makina.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Makina.dir/depend

