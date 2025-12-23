#!/bin/bash

currentdir=`dirname $0`
dir=$currentdir/../Environments/RaycastNav/Build/
configdir=$currentdir/../Environments/RaycastNav/Config/
config=${configdir}sac_mlagents.yaml
env_name=RaycastNav
env=${dir}${env_name}
name=mlagentsSAC_lstm

mlagents-learn $config --env=$env --run-id=$name --force --time-scale=20