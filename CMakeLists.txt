cmake_minimum_required(VERSION 3.23)
project(Makina C)

set(CMAKE_C_STANDARD 23)

IF(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3")
ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
ENDIF()

set(SOURCE_FILES src/main.c
        src/makina.h)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") # GCC / MinGW
    target_link_libraries(${PROJECT_NAME} -static-libgcc -static-libstdc++)
endif()

IF (WIN32)
    target_link_libraries(${PROJECT_NAME} -static winpthread)
ENDIF()