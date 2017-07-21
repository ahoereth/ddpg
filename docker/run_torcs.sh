#!/bin/bash

while true; do  # Restart torcs from time to time
  torcs -nolaptime -nofuel &  # add -d for debugging
  xte 'usleep 100000'
  xte 'key Return'
  xte 'usleep 100000'
  xte 'key Return'
  xte 'usleep 100000'
  xte 'key Up'
  xte 'usleep 100000'
  xte 'key Up'
  xte 'usleep 100000'
  xte 'key Return'
  xte 'usleep 100000'
  xte 'key Return'
  sleep 60
  pkill torcs
done

# tail -f $HOME/.vnc/*$DISPLAY.log  # Keep alive.
