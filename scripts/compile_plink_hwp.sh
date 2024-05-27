#!/usr/bin/env bash
gcc -g -fPIC -Wall -Werror -Wextra externals/plink_hwp.cc -shared -o fedalgo/gwasprs/plink_hwp.so
