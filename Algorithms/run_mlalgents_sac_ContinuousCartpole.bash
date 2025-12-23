#!/bin/bash

currentdir=`dirname $0`

env_name=ContinuousCartpole

envdir=$currentdir/../Environments/$env_name/
dir=${envdir}Build/Windows/
configdir=$envdir/Config/
config=${configdir}sac_mlagents.yaml
env=${dir}$env_name
name=mlagentsSAC_$env_name

mlagents-learn $config --env=$env --run-id=$name --force --time-scale=20