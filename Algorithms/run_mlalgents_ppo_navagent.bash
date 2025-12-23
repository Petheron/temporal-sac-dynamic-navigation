#!/bin/bash

currentdir=`dirname $0`
echo $currentdir
dir=$currentdir/../Environments/RaycastNav/Build/
configdir=$currentdir/../Environments/RaycastNav/Config/
echo $dir
config=${configdir}ppo.yaml
echo $config
env_name=RaycastNav
env=${dir}${env_name}
name=mlagentsPPO_walls

mlagents-learn $config --env=$env --run-id=$name --force --time-scale=50