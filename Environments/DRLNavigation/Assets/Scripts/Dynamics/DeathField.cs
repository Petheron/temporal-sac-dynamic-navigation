using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Serialization;

public class DeathField : Dynamic
{
    public EnvironmentProperties environmentProperties;
    private float yOffset;
    private EnvironmentParameters environmentParameters;

    protected override void Awake()
    {
        Initialize();
        base.Awake();
    }

    protected override void Start()
    {
        base.Start();
        SetYOffset(yOffset);
    }

    private void Initialize()
    {
        environmentParameters = Academy.Instance.EnvironmentParameters;
        GetProperties();
    }

    public void SetYOffset(float inputYOffset)
    {
        var t = transform;
        var position = t.position;
        position = new Vector3(position.x, inputYOffset, position.z);
        t.position = position;
        currentPosition = position;
        UpdateOccupancyNodes();
    }
    
    private void GetProperties()
    {
        yOffset = environmentParameters.GetWithDefault("death_field_y_offset", environmentProperties.DEATH_FIELD_Y_OFFSET);
    }
}
