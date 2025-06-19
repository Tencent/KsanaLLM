include(FetchContent)
set(TRPC_ROBUS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/trpc-robus)

FetchContent_Declare(
    trpc-robust
    GIT_REPOSITORY    https://git.woa.com/trpc-cpp/trpc-overload-control/trpc-robust.git
    GIT_TAG           feature/minchang_cmake_support
    SOURCE_DIR        ${TRPC_ROBUS_INSTALL_DIR}
)
FetchContent_MakeAvailable(trpc-robust)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/trpc-robust/include)
