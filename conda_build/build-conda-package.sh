#!/bin/bash

export CONDA_BLD_PATH=${CONDA_PREFIX}/conda-bld
pkg='pytorch-metric-learning'

conda build purge-all
conda-build -c pytorch $pkg

# convert package to other platforms
platforms=( osx-64 linux-32 win-32 win-64 )
find ${CONDA_BLD_PATH}/linux-64/ -name *.tar.bz2 | while read file
do
    echo $file
    for platform in "${platforms[@]}"
    do
       conda convert --platform $platform $file -o ${CONDA_BLD_PATH}
    done
    
done

# upload packages to conda
find ${CONDA_BLD_PATH} -name *.tar.bz2 | while read file
do
    echo $file
    anaconda upload $file
done

echo "Building conda package done!"