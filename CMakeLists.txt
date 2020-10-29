CMAKE_MINIMUM_REQUIRED (VERSION 3.13.4)
PROJECT (FFNeuralNetwork)

OPTION(WITHOUTEIGEN "Install module which works without Eigen submodule" ON)

SET (EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

CONFIGURE_FILE(NNConfig.h.in "${CMAKE_SOURCE_DIR}/include/NNConfig.h")

SET (PROJECT_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include")

INCLUDE(ExternalProject)
SET (LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")

SET (CMAKE_CXX_FLAGS " -O3 -msse2 -mssse3 -fexpensive-optimizations -std=c++11")
SET (CMAKE_C_FLAGS " -O3 -msse2 -mssse3 -fexpensive-optimizations -std=c++11")
    
SET (EIGEN_SRC_PATH ${CMAKE_SOURCE_DIR}/external/src/Eigen)

SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_SOURCE_DIR}/external)

if(NOT EXISTS "${EIGEN_SRC_PATH}") 
    ExternalProject_Add(
        Eigen
        GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
        # Disable install step
        INSTALL_COMMAND ""
        # Wrap download, configure and build steps in a script to log output
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON)
endif()

if (DEFINED INSTALL_PREFIX)
    message("INSTALLDIR set to " ${INSTALL_PREFIX})
    SET(CMAKE_INSTALL_PREFIX "${INSTALL_PREFIX}")
else (DEFINED INSTALL_PREFIX)
	message("INSTALLDIR set to default" ${CMAKE_INSTALL_PREFIX})
endif()


INCLUDE_DIRECTORIES("${PROJECT_INCLUDE_DIRS}" "${EIGEN_SRC_PATH}")
FIND_PACKAGE (Doxygen)
IF (DOXYGEN_FOUND)
    MESSAGE("Build documentation with: make docs")
    IF (EXISTS ${DOC_INPUT_FILE_PATH})
        MESSAGE("Existing File documentation with doxygen")
        configure_file(${DOC_INPUT_FILE_PATH}doxygen.conf ${DOC_OUTPUT_FILE_PATH}doxygen.conf @ONLY)
        configure_file(${DOC_INPUT_FILE_PATH}mainpage.dox ${DOC_OUTPUT_FILE_PATH}mainpage.dox @ONLY)
        ADD_CUSTOM_TARGET(
            docs
            ${DOXYGEN_EXECUTABLE} ${DOC_OUTPUT_FILE_PATH}doxygen.conf
            WORKING_DIRECTORY ${DOC_OUTPUT_FILE_PATH}
            COMMENT "Generating doxygen project documentation." VERBATIM
        )
    ELSE (EXISTS ${DOC_INPUT_FILE_PATH})
        ADD_CUSTOM_TARGET(docs COMMENT "Doxyfile not found. Please generate a doxygen configuration file to use this target." VERBATIM)
    ENDIF (EXISTS ${DOC_INPUT_FILE_PATH})
ELSE (DOXYGEN_FOUND)
    ADD_CUSTOM_TARGET(docs COMMENT "Doxygen not found. Please install doxygen to use this target." VERBATIM)
ENDIF (DOXYGEN_FOUND)
INCLUDE_DIRECTORIES("${PROJECT_INCLUDE_DIRS}" "${EIGEN_SRC_PATH}")


option(TESTS "Build the test" OFF)
if(TESTS)
   add_subdirectory(tests)
endif(TESTS)

add_subdirectory(examples)
