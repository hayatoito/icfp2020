#!/bin/sh

exec ./target/release/icfp2020 "$@" || echo "run error code: $?"
