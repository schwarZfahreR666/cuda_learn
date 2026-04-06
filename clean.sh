#!/bin/bash

PROJECT_SOURCE_DIR=$(cd "$(dirname "$0")" && pwd)

PROJECT_BUILD_DIR=$PROJECT_SOURCE_DIR"/build"

rm -rf $PROJECT_BUILD_DIR

rm -rf $PROJECT_SOURCE_DIR"/out"
