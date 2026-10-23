function(ompc_isolate_coverage_outputs)
  set(_root "${CMAKE_BINARY_DIR}/isolated")

  foreach(_target IN ITEMS
      omptarget
      LLVMOffload
      llvm-offload-mpi-proxy-device
      llvm-offload-device-info
      llvm-omp-kernel-replay
      offload-tblgen
      omptarget.rtl.mpi
      omptarget.rtl.cuda
      omptarget.rtl.host
      llvm_gtest
      llvm_gtest_main)

    if(TARGET "${_target}")
      set_target_properties("${_target}" PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${_root}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${_root}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${_root}/lib"
      )
      message(STATUS "OMPC isolation: ${_target}")
    endif()
  endforeach()

  if(TARGET libomptarget-nvptx)
    set_target_properties(libomptarget-nvptx PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${_root}/lib"
    )
    message(STATUS "OMPC isolation: libomptarget-nvptx")
  endif()

  if(TARGET omptarget.nvptx)
    set_target_properties(omptarget.nvptx PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${_root}/lib/nvptx64-nvidia-cuda"
    )
    message(STATUS "OMPC isolation: omptarget.nvptx")
  endif()

  if(TARGET omptarget.nvptx.all_objs)
    set_target_properties(omptarget.nvptx.all_objs PROPERTIES
      IMPORTED_OBJECTS "${_root}/lib/libomptarget-nvptx.bc"
    )
    message(STATUS "OMPC isolation: omptarget.nvptx.all_objs")
  endif()
endfunction()


cmake_language(DEFER CALL ompc_isolate_coverage_outputs)
