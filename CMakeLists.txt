cmake_minimum_required(VERSION 3.16)
project(spatial_viz)

set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(NOMINMAX)

add_subdirectory(external/glfw)

set(IMGUI_DIR external/imgui)

add_library(imgui
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    external/glfw/include
)

add_library(glad external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

# Make sure the compiler can find audio_engine.h and miniaudio.h
include_directories(external)

add_executable(app 
    main.cpp
    external/FFTConvolver.cpp
    external/AudioFFT.cpp
    external/Utilities.cpp
)

find_package(OpenGL REQUIRED)

target_link_libraries(app
    glfw
    imgui
    glad
    OpenGL::GL
)

# Link standard libraries required by miniaudio on Linux/macOS
if (UNIX AND NOT APPLE)
    target_link_libraries(app m pthread dl)
endif()