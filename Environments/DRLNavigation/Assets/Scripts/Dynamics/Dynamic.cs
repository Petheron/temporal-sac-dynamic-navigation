using System;
using System.Collections.Generic;
using UnityEngine;


public class Dynamic : OccupancyMapObservable
{
    public bool isDynamic = true;
    protected bool hasPositionChanged;

    private int defaultLayer;

    public bool test;

    public event Action<Vector3> OnPositionUpdate;

    protected override void Awake()
    {
        defaultLayer = gameObject.layer; 
        base.Awake();
    }

    public void SetDynamicState(bool newDynamicState)
    {
        isDynamic = newDynamicState;
        Reset();
    }

    protected virtual void FixedUpdate()
    {
        if (isDynamic)
        {
            PerformContinuousDynamics();
        }

        if (hasPositionChanged)
        {
            UpdateOccupancyNodes();
            OnPositionUpdate?.Invoke(currentPosition);
            hasPositionChanged = false;
        }

        if (test)
        {
            Test();
            test = false;
        }
    }

    public override void Reset()
    {
        base.Reset();
        SetLayer();
    }

    private void SetLayer()
    {
        gameObject.layer = isDynamic ? defaultLayer : 0;
    }

    protected virtual void PerformContinuousDynamics()
    {
        return;
    }

    protected virtual void PerformDiscreteDynamics(Collider other)
    {
        if (!isDynamic) return;
        if (other.isTrigger) return;
    }

    protected virtual void OnTriggerEnter(Collider other)
    {
        PerformDiscreteDynamics(other);
    }

    protected virtual void Test()
    {
        return;
    }
}
