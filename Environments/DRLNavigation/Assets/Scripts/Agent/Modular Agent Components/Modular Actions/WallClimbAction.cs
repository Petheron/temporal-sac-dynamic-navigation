
using System;
using System.Collections.Generic;
using UnityEngine;

public class WallClimbAction:PassiveAgentAction
{
    private enum MoveDirection
    {
        Forward = 0,
        Strafe = 1
    }
    private readonly MoveDirection[] moveDirections = new[] {MoveDirection.Forward, MoveDirection.Strafe};
    
    private float wallClimbSensitivity;
    private float wallVerticalSensitivity;
    private float wallClimbCheckDistance;
    private float wallClimbFallSpeedModifier=0.0001f;
    [SerializeField] private float wallClimbDuration = 1f;

    private float acceleration;
    private float wallClimbAcceleration;
    private LayerMask layerMask;

    private ActionTimer wallClimbTimer;

    [SerializeField, ReadOnly] private bool isWallClimbing;
    
    private float[] inputs;
    private Vector3[] movementVectors;
    private Vector3 wallNormal;

    private float Acceleration => movement.currentMutator.AccelerationModifier * acceleration;
    private float WallClimbAcceleration => movement.currentMutator.AccelerationModifier * wallClimbAcceleration;
    private bool IsGrounded => movement.IsGrounded;
    private bool RestartWallClimbTimer => !isWallClimbing && wallClimbTimer.HasStarted && IsGrounded;
    
    public override void InitializeAgentProperties()
    {
        acceleration = agentProps.MOVE_ACCELERATION;
        wallClimbAcceleration = agentProps.WALLCLIMB_ACCELERATION;
        wallClimbSensitivity = agentProps.WALLCLIMB_SENSITIVITY;
        wallVerticalSensitivity = agentProps.WALL_VERTICALITY_SENSITIVITY;
        wallClimbCheckDistance = agentProps.WALL_CHECK_DISTANCE;
        wallClimbFallSpeedModifier = agentProps.WALLCLIMB_FALL_SPEED_MODIFIER;
        wallClimbDuration = agentProps.WALLJUMP_DURATION;
        layerMask = agentProps.GROUNDED_MASK;
    }
    
    public override void InitializeEnvironmentParameters()
    {
        usePassiveAction = envParams.GetWithDefault("use_wallclimb", usePassiveAction ? 1 : 0) > 0;
    }

    protected override void InitializeActionTimers()
    {
        wallClimbTimer = new ActionTimer(wallClimbDuration);
        
    }

    public override void ProcessActionInput(float value)
    {
        ProcessActionInput(value,0f);
    }

    // Receives pre-processed inputs from MoveAction
    public override void ProcessActionInput(float value1, float value2)
    {
        inputs[0] = value1;
        inputs[1] = value2;
    }

    public override List<float> CollectObservations()
    {
        if (!usePassiveAction) return new List<float>();
        return new List<float>()
        {
            Convert.ToSingle(isWallClimbing),
            wallClimbTimer.GetRemainingTime(true)
        };
    }


    public override bool IsActive()
    {
        if (!usePassiveAction) return false;
        if (wallClimbTimer.IsFinished)
        {
            EndWallClimb();
            return false;
        }
        StoreWallNormal();
        if (wallNormal == Vector3.zero)
        {
            EndWallClimb();
            return false;
        }
        
        if(!isWallClimbing) StartWallClimb();
        return true;
    }

    private void StoreWallNormal()
    {
        wallNormal = Vector3.zero;
        if (!Physics.Raycast(transform.position, transform.forward, out var hit, 
                wallClimbCheckDistance, layerMask, QueryTriggerInteraction.Ignore)) return;
        if (Vector3.Dot(hit.normal, transform.forward) >= -wallClimbSensitivity) return;
        if (Mathf.Abs(Vector3.Dot(hit.normal,Vector3.up))>wallVerticalSensitivity) return;
        wallNormal = hit.normal;
    }

    private void StartWallClimb()
    {
        if (isWallClimbing) return;
        movement.fallingMultiplier *= wallClimbFallSpeedModifier;
        isWallClimbing = true;
    }

    private void EndWallClimb()
    {
        if (!isWallClimbing) return;
        wallNormal = Vector3.zero;
        movement.fallingMultiplier /= wallClimbFallSpeedModifier;
        isWallClimbing = false;
    }

    public override void Perform()
    {
        if (!usePassiveAction) return;
        if (!isWallClimbing) return;
        CalculateMovementVectors();
        Move();
    }
    
    private void CalculateMovementVectors()
    {
        // Calculate the movement direction (accommodates for sloped surfaces)

        foreach (var direction in moveDirections)
        {
            Vector3 transformDirection = direction == MoveDirection.Strafe ? -transform.forward : transform.right;
            Vector3 movementVector = Vector3.Cross(transformDirection, movement.groundNormal);
            
            movementVectors[(int) direction] = movementVector;
        }
    }

    private void Move()
    {
        if (!isWallClimbing) return;
        Vector3 moveVector = Vector3.zero;

        foreach (var direction in moveDirections)
        {
            moveVector += CalculateForceMovement(direction);
        }
        agentRigidbody.AddForce(moveVector, ForceMode.Acceleration);
    }

    private Vector3 CalculateForceMovement(MoveDirection direction)
    {
        Vector3 movementVector = movementVectors[(int) direction];
        float input = inputs[(int) direction];
        float magnitude = input * Acceleration;

        if (direction == MoveDirection.Forward && input>0f)
        {
            movementVector = Vector3.Cross(movementVectors[(int) MoveDirection.Strafe], wallNormal);
            magnitude = WallClimbAcceleration;

            // remove negative y-vel if present
            var v = agentRigidbody.velocity;
            if (v.y < 0f)
            {
                v.y = 0f;
                agentRigidbody.velocity = v;
            }
        }

        return magnitude*movementVector;
    }

    private void FixedUpdate()
    {
        if(!usePassiveAction) return;
        if (RestartWallClimbTimer)
        {
            wallClimbTimer.Start();
            return;
        }
        if(isWallClimbing) wallClimbTimer.Decrement(Time.fixedDeltaTime);
    }

    public override void Reset()
    {
        inputs = new []{0f,0f};
        movementVectors = new[] {Vector3.zero, Vector3.zero};
        wallClimbTimer.Reset();
    }
}
