using System;
using System.Collections.Generic;
using UnityEngine;

public class GrappleAction : AgentAction
{
    [Header("Grapple Parameters")]
    [SerializeField] private float grappleDistance;
    [SerializeField] private float grappleDuration;
    [SerializeField] private float grappleCooldown;
    private ActionTimer grappleDurationTimer;
    private ActionTimer grappleCooldownTimer;
    private LayerMask grappleLayerMask;

    private bool isGrappleInputPressed;
    private Vector3 startPosition;
    private Vector3 endPosition;
    private Vector3 grappleDirection;

    private bool CanGrapple => !IsGrappleOnCooldown && !IsGrappling;
    private bool IsGrappleOnCooldown => !grappleCooldownTimer.IsFinished;
    private bool IsGrappling =>  !grappleDurationTimer.IsFinished;

    private float Velocity => grappleDistance / grappleDuration;

    public override void InitializeAgentProperties()
    {
        if(!agentProps) return;
        inputSize = 1;
        grappleDistance = agentProps.GRAPPLE_DISTANCE;
        grappleDuration = agentProps.GRAPPLE_DURATION;
        grappleCooldown = agentProps.GRAPPLE_COOLDOWN;
        grappleLayerMask = agentProps.GROUNDED_MASK;
    }

    protected override void InitializeActionTypes()
    {
        actionTypes.Add(ModularActionType.Grapple);
        complexActionType = ComplexActionType.Grapple;
    }

    protected override void InitializeActionTimers()
    {
        grappleCooldownTimer = new ActionTimer(grappleCooldown);
        grappleDurationTimer = new ActionTimer(grappleDuration);

        grappleDurationTimer.OnFinished += EndGrapple;
    }

    public override void ProcessActionInput(float value)
    {
        isGrappleInputPressed = BinarizeBooleanInputValue(value);
    }

    public override void Perform()
    {
        Grapple();
    }

    private void Grapple()
    {
        if (!isGrappleInputPressed) return;
        if (!CanGrapple) return;
        CastGrapple();
        
        StorePosition();
    }

    private void CastGrapple()
    {
        startPosition = agentRigidbody.position;
        grappleDirection = cameraRotationTransform.forward;

        if (Physics.Raycast(startPosition, grappleDirection, out var hit, grappleDistance, grappleLayerMask,
                QueryTriggerInteraction.Ignore))
        {
            endPosition = hit.point;
            StartGrapple();
        }
        else
        {
            EndGrapple();
        }
    }

    private void StartGrapple()
    {
        movement.canTakeActions = false;
        movement.gravityEnabled = false;

        agentRigidbody.velocity = Velocity * grappleDirection;
        grappleDurationTimer.Start();
    }
    
    private void EndGrapple()
    {
        movement.canTakeActions = true;
        movement.gravityEnabled = true;
        agentRigidbody.velocity = Vector3.zero;
        grappleDurationTimer.Reset();
        grappleCooldownTimer.Start();
    }

    private void Update()
    {
        grappleCooldownTimer?.Decrement(Time.deltaTime);
        grappleDurationTimer?.Decrement(Time.deltaTime);
    }

    public override List<float> CollectObservations()
    {
        List<float> obs = new List<float>
        {
            Convert.ToSingle(CanGrapple),
            Convert.ToSingle(IsGrappling)
        };
        return obs;
    }

    public override void Reset()
    {
        grappleCooldownTimer.Reset();
        grappleDurationTimer.Reset();
        isGrappleInputPressed = false;
        startPosition = Vector3.zero;
        endPosition = Vector3.zero;
        
        actionUsedPositions = new List<Vector3>();
    }
}
