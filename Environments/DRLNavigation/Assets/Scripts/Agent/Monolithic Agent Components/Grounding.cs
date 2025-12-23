using System;
using UnityEngine;


public class Grounding:MonoBehaviour
{
    private BaseMovement movement;
    private Rigidbody groundRigidbody;
    public Vector3 Velocity => groundRigidbody ? groundRigidbody.velocity : Vector3.zero;

    private void Awake()
    {
        movement = GetComponentInParent<BaseMovement>();
    }

    private void StoreRigidbody(Collider other)
    {
        if(!groundRigidbody) groundRigidbody = other.attachedRigidbody;
    }
    
    private void ReleaseRigidbody(Collider other)
    {
        if (groundRigidbody == other.attachedRigidbody) groundRigidbody = null;
    }

    private void OnTriggerEnter(Collider other)
    {
        StoreRigidbody(other);
        if (!movement) return;
        movement.AttachGroundRigidbody(groundRigidbody);
    }

    private void OnTriggerStay(Collider other)
    {
        StoreRigidbody(other);
    }

    private void OnTriggerExit(Collider other)
    {
        ReleaseRigidbody(other);
        if (!movement) return;
        movement.DetachGroundRigidBody(other.attachedRigidbody);
    }
}
