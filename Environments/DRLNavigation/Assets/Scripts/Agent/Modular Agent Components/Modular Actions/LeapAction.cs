using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

public class LeapAction : AgentAction
{
    [Header("Leap Parameters")] 
    [SerializeField] private float leapVelocity;
    [SerializeField] private float leapAngleMin;

    private bool isInputPressed;
    private bool hasPerformedLeap;

    private bool CanLeap => !hasPerformedLeap;
    private bool IsGrounded => movement.IsGrounded;
    private float LeapCosAngleMin => Mathf.Cos(Mathf.Deg2Rad * (90f-Mathf.Clamp(leapAngleMin,0f,90f)));
    private float leapCosAngleMin;
    

    public override void InitializeAgentProperties()
    {
        if(!ModularAgentProps) return;
        inputSize = 1;
        leapVelocity = ModularAgentProps.LEAP_VELOCITY;
        leapAngleMin = ModularAgentProps.LEAP_ANGLE_MINIMUM;
        leapCosAngleMin = LeapCosAngleMin;
    }

    public override void InitializeEnvironmentParameters()
    { }
    
    protected override void InitializeActionTypes()
    {
        actionTypes.Add(ModularActionType.Leap);
        complexActionType = ComplexActionType.Leap;
    }

    protected override void InitializeActionTimers()
    { }

    public override void ProcessActionInput(float value)
    {
        isInputPressed = BinarizeBooleanInputValue(value); 
    }

    public override void Perform()
    {
        Leap();
    }

    private void Leap()
    {
        if (!isInputPressed) return;
        if (!CanLeap) return;

        hasPerformedLeap = true;
        
        ApplyForce();
        
        StorePosition();
    }

    private void ApplyForce()
    {
        Vector3 direction = cameraRotationTransform.forward;
        if (direction.y < leapCosAngleMin)
        {
            direction = transform.forward;
            direction.y = leapCosAngleMin;
        }

        // remove vertical velocity component and horizontal velocity in same direction as leap.
        var velocity = agentRigidbody.velocity;
        velocity -= Vector3.Scale(new Vector3(Mathf.Abs(direction.x),0f,Mathf.Abs(direction.z)), velocity);
        velocity.y = 0f;
        agentRigidbody.velocity = velocity;
        
        // remove accumulated force from other actions before this one (especially jumping)
        agentRigidbody.AddForce(-1.0f*agentRigidbody.GetAccumulatedForce(),ForceMode.Force);
        
        agentRigidbody.AddForce(leapVelocity*direction, ForceMode.VelocityChange);
    }
    
    private void FixedUpdate()
    {
        if (!movement) return;

        if (IsGrounded && hasPerformedLeap && agentRigidbody.velocity.y<=0f)
        {
            hasPerformedLeap = false;
        }
    }

    public override List<float> CollectObservations()
    {
        List<float> obs = new List<float>
        {
            Convert.ToSingle(CanLeap)
        };
        return obs;
    }

    public override void Reset()
    {
        hasPerformedLeap = true;
        isInputPressed = false;
        actionUsedPositions = new List<Vector3>();
    }
    
#if UNITY_EDITOR
    private void OnValidate()
    {
        leapCosAngleMin = LeapCosAngleMin;
    }
#endif
}
