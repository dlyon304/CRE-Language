#!/bin/bash

iterations=12
done=0

i=0
while [ True ]
do
    if [ $i -eq $iterations ]
    then
        i=Last
        done=1
    fi
    echo $i

    if [ $done -eq 1 ]
    then
        break
    fi
    ((i++))
done 



