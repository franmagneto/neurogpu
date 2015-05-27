#!/bin/bash

for i in {1..10}
do
	src/neurogpu-plus $@ > /dev/null
done
