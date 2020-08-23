rm CMakeCache.txt
cmake ../ \
    -DCMAKE_INSTALL_PREFIX=/project/rrg-nadaraja-ac/Libraries/Trilinos/install-dealii               \
    -DCMAKE_BUILD_TYPE=RELEASE                                                                      \
    -DCMAKE_CXX_COMPILER=mpicxx                                                                     \
    -DCMAKE_C_COMPILER=mpicc                                                                        \
    -DTrilinos_ENABLE_Fortran=OFF                                                                   \
    -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF                                                     \
    -DTPL_ENABLE_MPI=ON                                                                             \
    -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON                                                     \
    -DTrilinos_ENABLE_Amesos=ON                                                                     \
    -DTrilinos_ENABLE_Epetra=ON                                                                     \
    -DTrilinos_ENABLE_EpetraExt=ON                                                                  \
    -DTrilinos_ENABLE_Ifpack=ON                                                                     \
    -DTrilinos_ENABLE_AztecOO=ON                                                                    \
    -DTrilinos_ENABLE_Sacado=ON                                                                     \
    -DTrilinos_ENABLE_Teuchos=ON                                                                    \
    -DTrilinos_ENABLE_ML=ON                                                                         \
    -DTrilinos_ENABLE_ROL=ON                                                                        \
    -DTrilinos_ENABLE_Tpetra=ON                                                                     \
    -DTrilinos_ENABLE_Zoltan=ON                                                                     \
    -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON \
    -DTrilinos_ENABLE_COMPLEX_FLOAT=ON \
    -DTrilinos_VERBOSE_CONFIGURE=OFF                                                                \
    -DBUILD_SHARED_LIBS=ON                                                                          \
    -DCMAKE_VERBOSE_MAKEFILE=OFF                                                                    \
    -DTPL_ENABLE_HDF5=ON                                                                            \
    -DHDF5_INCLUDE_DIRS=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/MPI/intel2018.3/openmpi3.1/hdf5-mpi/1.10.3/include \
    -DHDF5_LIBRARY_DIRS=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/MPI/intel2018.3/openmpi3.1/hdf5-mpi/1.10.3/lib \
    -DTPL_ENABLE_BLAS=ON                                                                            \
    -DBLAS_LIBRARY_NAMES='openblas'                                                                 \
    -DBLAS_INCLUDE_DIRS=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/gcc7.3/openblas/0.3.4/include \
    -DBLAS_LIBRARY_DIRS=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/gcc7.3/openblas/0.3.4/lib \
    -DTPL_LAPACK_INCLUDE_DIRS=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/gcc7.3/openblas/0.3.4/include \
    -DTPL_LAPACK_LIBRARIES=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/gcc7.3/openblas/0.3.4/lib/libopenblas.so \
    -DTPL_ENABLE_Boost=ON                                                                           \
    -DBoost_INCLUDE_DIRS=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/gcc7.3/boost/1.68.0/include \
    -DBoost_LIBRARY_DIRS=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/gcc7.3/boost/1.68.0/lib \