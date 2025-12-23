using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JumpPad : Dynamic
{

    public float pushForce = 10f;

    public override void Reset()
    {
        base.Reset();
        gameObject.SetActive(isDynamic);
    }


    protected override void PerformDiscreteDynamics(Collider other)
    {
        base.PerformDiscreteDynamics(other);
        var rigidBody = other.attachedRigidbody;
        if (rigidBody)
        {   
            Vector3 vel = rigidBody.velocity;
            vel.y = 0f;
            rigidBody.velocity = vel;
            
            rigidBody.AddForce(-rigidBody.GetAccumulatedForce().y*Vector3.up, ForceMode.Force);
            
            rigidBody.AddForce(transform.up*pushForce,ForceMode.VelocityChange);
        }
    }
}
