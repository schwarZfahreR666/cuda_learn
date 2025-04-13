#!/bin/bash

PROJECT_SOURCE_DIR=$(cd "$(dirname "$0")" && pwd)

PROJECT_BUILD_DIR=$PROJECT_SOURCE_DIR"/build"

cd $PROJECT_SOURCE_DIR

if [ ! -d $PROJECT_BUILD_DIR ];
then
    mkdir $PROJECT_BUILD_DIR
fi

if [ ! -d $PROJECT_SOURCE_DIR"/out" ];
then
    mkdir $PROJECT_SOURCE_DIR"/out"
fi

cd $PROJECT_BUILD_DIR

cmake .. && make

# 拷贝无扩展名的可执行文件
find "$PROJECT_BUILD_DIR" -type f -executable ! -name "*.*" -exec cp {} $PROJECT_SOURCE_DIR"/out/" \;



