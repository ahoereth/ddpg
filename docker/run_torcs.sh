#!/bin/bash

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
xte 'usleep 100000'
xte 'key F2'  # Change view
xte 'key F2'
xte 'key F2'
xte 'usleep 100000'
xte "keydown Alt_L" "key F11" "keyup Alt_L" # Full screen

# Kill container after some time, docker will automatically restart it.
# Helps to prevent torcs memory leak.
sleep 1800  # Every 30 minutes.
shutdown -h now  # Shutdown container.

# tail -f $HOME/.vnc/*$DISPLAY.log  # Keep alive.
