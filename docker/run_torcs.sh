#!/bin/bash

torcs -nolaptime -nofuel &  # add -d for debugging

# Start training race
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

# Change to close first person view
# xte 'usleep 100000'
# xte 'key F2'
# xte 'key F2'
# xte 'key F2'

# Change to full screen
xte 'usleep 100000'
xte "keydown Alt_L" "key F11" "keyup Alt_L"

# Kill container after some time, docker will automatically restart it.
# Helps to prevent torcs memory leak.
sleep 1800  # Every 30 minutes.
shutdown -h now  # Shutdown container.

# tail -f $HOME/.vnc/*$DISPLAY.log  # Keep alive.
