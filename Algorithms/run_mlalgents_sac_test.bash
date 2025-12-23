#!/bin/bash

currentdir=`dirname $0`
#envName=3DBall
envName=ContinuousCartpole

dir=$currentdir/../Environments/$envName/Build/

configdir=$currentdir/../Environments/$envName/Config/
config=${configdir}sac.yaml

env=${dir}$envName
name=mlagentsSAC

mlagents-learn $config --env=$env --run-id=$name --force --time-scale=50 --no-graphics