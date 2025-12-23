using System;
using System.Collections.Generic;
using LevelGeneration;
using Unity.MLAgents;
using UnityEngine;

public class Level:MonoBehaviour
{
    private NavMeshConstructor navMeshConstructor;
    private LevelGenerator levelGenerator;
    private LevelDynamicsManager levelDynamicsManager;

    private AgentProperties agentProperties;
    private EnvironmentParameters environmentParameters;
    
    private void Awake()
    {
        environmentParameters = Academy.Instance.EnvironmentParameters;
        levelDynamicsManager = GetComponent<LevelDynamicsManager>();
        navMeshConstructor = GetComponent<NavMeshConstructor>();
        levelGenerator = GetComponentInChildren<LevelGenerator>();
    }

    public void Initialize(EnvironmentParameters environmentParams, AgentProperties agentProps, OccupancyMap occupancyMap, CameraController cameraController)
    {
        environmentParameters = environmentParams;
        agentProperties = agentProps;

        // Setup camera controller
        if (cameraController)
        {
            foreach (var cam in transform.GetComponentsInChildren<Camera>())
            {
                cameraController.AddCamera(cam);
            }
        }
        
        // Setup level dynamics
        if (levelDynamicsManager)
        {
            bool useDynamics = environmentParameters.GetWithDefault("use_dynamics", agentProperties.USE_DYNAMICS?1.0f:0f) > 0f;
            levelDynamicsManager.SetLevelDynamics(useDynamics);
        }
        
        // Setup level generator and build its NavMesh or just build the NavMesh
        if (levelGenerator)
        {
            levelGenerator.Initialize(environmentParams, occupancyMap);
        }
        else
        {
            if (navMeshConstructor)
            {
                navMeshConstructor.BuildNavMesh();
            }
        }
    }
}
