#!/bin/bash

currentdir=`dirname $0`

envName=3DBall

dir=$currentdir/../Environments/$envName/Build/
configdir=$currentdir/../Environments/$envName/Config/

config=${configdir}ppo.yaml

env=${dir}$envName
name=mlagentsPPO

mlagents-learn $config --env=$env --run-id=$name --force --time-scale=20