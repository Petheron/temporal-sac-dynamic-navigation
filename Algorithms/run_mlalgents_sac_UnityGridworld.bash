#!/bin/bash

currentdir=`dirname $0`

env_name=UnityGridworld

envdir=$currentdir/../Environments/$env_name/
dir=$envdir/Build/
configdir=$envdir/Config/
config=${configdir}sac.yaml
env=${dir}$env_name
name=mlagentsSAC_$env_name

mlagents-learn $config --env=$env --run-id=$name --force --time-scale=20