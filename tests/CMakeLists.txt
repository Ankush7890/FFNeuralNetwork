# Enable ExternalProject CMake module
INCLUDE(ExternalProject)

# Set default ExternalProject root directory
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)

# Add gtest
ExternalProject_Add(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG release-1.6.0
    # Disable install step
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

    

# Specify include dir
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIR ${source_dir}/include)

# Library
ExternalProject_Get_Property(googletest binary_dir)
set(GTEST_LIBRARY_PATH ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_LIBRARY gtest)
add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION
                ${GTEST_LIBRARY_PATH} )
add_dependencies(${GTEST_LIBRARY} googletest)

# build and run neural network from scratch tests
INCLUDE_DIRECTORIES("${source_dir}/include")
FILE (GLOB_RECURSE test_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)
SET (test_LIBS ${GTEST_LIBRARY} pthread dl)
SET (test_BIN ${PROJECT_NAME}-tests)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/tests)

configure_file(iris.csv tests/iris.csv COPYONLY)

IF (NOT CMAKE_CROSSCOMPILING)
    LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
    ADD_EXECUTABLE(${test_BIN} ${test_SRCS})
    TARGET_LINK_LIBRARIES(${test_BIN} ${test_LIBS})
    MESSAGE("Running tests...")
    ADD_CUSTOM_TARGET(check ALL "${CMAKE_BINARY_DIR}/tests/${test_BIN}" DEPENDS ${test_BIN} COMMENT "Executing tests..." VERBATIM SOURCES ${test_SRCS})
    ADD_CUSTOM_TARGET(testrun "${CMAKE_BINARY_DIR}/tests/${test_BIN}" DEPENDS ${test_BIN} COMMENT "Executing tests..." VERBATIM SOURCES ${test_SRCS})
ENDIF (NOT CMAKE_CROSSCOMPILING)
