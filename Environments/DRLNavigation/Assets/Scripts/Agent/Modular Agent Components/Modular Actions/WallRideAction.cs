using System;
using System.Collections.Generic;
using UnityEngine;

public class WallRideAction:PassiveAgentAction
{
     private enum MoveDirection
    {
        Forward = 0,
        Strafe = 1
    }
    private readonly MoveDirection[] moveDirections = new[] {MoveDirection.Forward, MoveDirection.Strafe};

    [SerializeField, ReadOnly] private bool isWallRiding;
    [SerializeField] private float wallRideAccelModifier = 1.0f;
    [SerializeField] private float wallRideFallSpeedModifier = 0.25f;
    [SerializeField] private float wallRideGravityModifier = 0.25f;
    [SerializeField] private float wallRideDragModifier = 0.5f;
    
    private float wallVerticalSensitivity;
    private float wallCheckDistance;

    private float acceleration;
    private LayerMask layerMask;


    private float[] inputs;
    private Vector3[] movementVectors;
    [SerializeField, ReadOnly] private Vector3 wallNormal;
    [SerializeField, ReadOnly] private Vector3 wallRideVelocity;
    
    private bool IsGrounded => movement.IsGrounded;

    private float Acceleration => movement.currentMutator.AccelerationModifier * wallRideAccelModifier* acceleration;

    private bool IsWallVertical(Vector3 normal) => 1f - Mathf.Abs(Vector3.Dot(Vector3.up, normal)) >= wallVerticalSensitivity;
    
    public override void InitializeAgentProperties()
    {
        acceleration = agentProps.MOVE_ACCELERATION;
        wallVerticalSensitivity = agentProps.WALL_VERTICALITY_SENSITIVITY;
        wallCheckDistance = agentProps.WALL_CHECK_DISTANCE;
        layerMask = agentProps.GROUNDED_MASK;

        wallRideAccelModifier = agentProps.WALLRIDE_ACCELERATION_MODIFIER;
        wallRideFallSpeedModifier = agentProps.WALLRIDE_FALL_SPEED_MODIFIER;
        wallRideGravityModifier = agentProps.WALLRIDE_GRAVITY_MODIFIER;
        wallRideDragModifier = agentProps.WALLRIDE_DRAG_MODIFIER;
    }

    public override void InitializeEnvironmentParameters()
    {
        usePassiveAction = envParams.GetWithDefault("use_wallride", usePassiveAction ? 1 : 0) > 0;
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
        var obs = new List<float>()
        {
            Convert.ToSingle(isWallRiding),
        };
        obs.AddRange(AgentUtility.VectorToList(wallRideVelocity.normalized));
        return obs;
    }


    public override bool IsActive()
    {
        // always return false if not used
        if (!usePassiveAction) return false;
        
        if (IsGrounded)
        {
            EndWallRide();
            return false;
        }

        StoreWallNormal();
        
        if (wallNormal == Vector3.zero)
        {
            EndWallRide();
            return false;
        }

        if(!isWallRiding) StartWallRide();

        return true;
    }

    public override void Perform()
    {
        if (!usePassiveAction) return;
        if (!isWallRiding) return;
        AlignTransformToWallNormal();
        Move();
    }

    private void AlignTransformToWallNormal()
    {
        wallRideVelocity = Vector3.ProjectOnPlane(agentRigidbody.velocity, wallNormal);
        agentRigidbody.velocity = wallRideVelocity;
        
    }

    private void StoreWallNormal()
    {
        wallNormal = Vector3.zero;
        float wallDistance = Mathf.Infinity;
        var t = transform;
        int wallSide = 0;
        if (Physics.Raycast(t.position, t.right, out var rightHit,
                wallCheckDistance, layerMask, QueryTriggerInteraction.Ignore))
        {
            if (rightHit.distance < wallDistance && IsWallVertical(rightHit.normal))
            {
                wallDistance = rightHit.distance;
                wallNormal = rightHit.normal;
                wallSide = -1;
            }
        }

        if (Physics.Raycast(t.position, -t.right, out var leftHit,
                wallCheckDistance, layerMask, QueryTriggerInteraction.Ignore))
        {
            if (leftHit.distance < wallDistance && IsWallVertical(leftHit.normal))
            {
                wallDistance = leftHit.distance;
                wallNormal = leftHit.normal;
                wallSide = 1;
            }
        }

        if (wallSide != 0)
        {
            movementVectors[(int) MoveDirection.Strafe] = wallSide * wallNormal;
            movementVectors[(int) MoveDirection.Forward] = Vector3.Cross(movementVectors[(int) MoveDirection.Strafe], Vector3.up);
            
            transform.forward = movementVectors[(int) MoveDirection.Forward];
            movement.SetRotation(movement.Rotation);
        }
        
    }

    private void StartWallRide()
    {
        if (isWallRiding) return;
        movement.fallingMultiplier *= wallRideFallSpeedModifier;
        movement.gravityModifier *= wallRideGravityModifier;
        movement.dragModifier *= wallRideDragModifier;
        isWallRiding = true;
        movement.rotateTransform = false;
    }

    private void EndWallRide()
    {
        if (!isWallRiding) return;

        isWallRiding = false;
        movement.fallingMultiplier /= wallRideFallSpeedModifier;
        movement.dragModifier /= wallRideDragModifier;
        movement.gravityModifier /= wallRideGravityModifier;
        wallNormal = Vector3.zero;
        wallRideVelocity = Vector3.zero;
        movement.rotateTransform = true;
        movement.SetRotation(movement.Rotation);
    }

    private void Move()
    {
        if (!isWallRiding) return;
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

        return magnitude*movementVector;

    }

    public override void Reset()
    {
        inputs = new []{0f,0f};
        movementVectors = new[] {Vector3.zero, Vector3.zero};
        wallRideVelocity = Vector3.zero;
        wallNormal = Vector3.zero;
    }
}
