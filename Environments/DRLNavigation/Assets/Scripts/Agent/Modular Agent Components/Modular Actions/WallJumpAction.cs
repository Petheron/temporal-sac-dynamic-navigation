using System;
using System.Collections.Generic;
using UnityEngine;

public class WallJumpAction:PassiveAgentAction
{
    // walljump combines with existing jump action rather than overriding it completely.
    
    private enum WallJumpCheckDirections
    {
        Front,
        Back,
        Left,
        Right
    }

    [SerializeField] private List<WallJumpCheckDirections> wallJumpCheckDirections;
    [SerializeField] private float wallJumpVelocity = 10f;
    [SerializeField, ReadOnly] private bool canWallJump;
    
    private float wallCheckDistance = 0.6f;
    private float wallVerticalSensitivity = 1.0f;
    private LayerMask layerMask;
    [SerializeField, ReadOnly] private Vector3 wallNormal;
    
    private bool IsWallVertical(Vector3 normal) => 1f - Mathf.Abs(Vector3.Dot(Vector3.up, normal)) >= wallVerticalSensitivity;
    
    public override void InitializeAgentProperties()
    {
        wallJumpVelocity = agentProps.WALLJUMP_VELOCITY;
        wallVerticalSensitivity = agentProps.WALL_VERTICALITY_SENSITIVITY;
        wallCheckDistance = agentProps.WALL_CHECK_DISTANCE;
        layerMask = agentProps.GROUNDED_MASK;
    }
    
    public override void InitializeEnvironmentParameters()
    {
        usePassiveAction = envParams.GetWithDefault("use_walljump", usePassiveAction ? 1 : 0) > 0;
    }

    public override bool IsActive()
    {
        canWallJump = false;
        if (!usePassiveAction) return false;
        if (movement.IsGrounded)
        {
            wallNormal = Vector3.zero;
            return false;
        }
        GetWallNormal();
        if (wallNormal == Vector3.zero) return false;
        canWallJump = true;
        return true;
    }

    public override void Perform()
    {
        if (!usePassiveAction) return;
        if (!canWallJump) return;
        
        agentRigidbody.AddForce(wallJumpVelocity*wallNormal, ForceMode.VelocityChange);
    }

    private void GetWallNormal()
    {
        float closestDistance = Mathf.Infinity;
        wallNormal = Vector3.zero;

        foreach (var direction in wallJumpCheckDirections)
        {
            Vector3 rayDirection = Vector3.zero;
            switch (direction)
            {
                case WallJumpCheckDirections.Front:
                    rayDirection = transform.forward;
                    break;
                case WallJumpCheckDirections.Back:
                    rayDirection = -transform.forward;
                    break; 
                case WallJumpCheckDirections.Left:
                    rayDirection = -transform.right;
                    break;
                case WallJumpCheckDirections.Right:
                    rayDirection = transform.right;
                    break; 
            }

            if (rayDirection == Vector3.zero) continue;
            if (!Physics.Raycast(transform.position, rayDirection, out var hit,
                    wallCheckDistance, layerMask, QueryTriggerInteraction.Ignore)) continue;
            if (!IsWallVertical(hit.normal)) continue;
            if (hit.distance < closestDistance - Mathf.Epsilon)
            {
                closestDistance = hit.distance;
                wallNormal = hit.normal;
            }
            else if (Mathf.Abs(hit.distance-closestDistance) < Mathf.Epsilon)
            {
                wallNormal += hit.normal;
                wallNormal.Normalize();
            }
        }
    }

    public override List<float> CollectObservations()
    {
        if (!usePassiveAction) return new List<float>();
        var obs = new List<float>();
        obs.Add(Convert.ToSingle(canWallJump));
        obs.AddRange(AgentUtility.VectorToList(wallNormal));
        return obs;
    }

    public override void Reset()
    {
        canWallJump = false;
        wallNormal = Vector3.zero;
    }
}
