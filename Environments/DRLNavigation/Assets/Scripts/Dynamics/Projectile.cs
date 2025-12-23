using System;
using UnityEngine;

public class Projectile : Dynamic
{
    public float speed = 10f;
    public float lifeTime = 5f;
    private float timer;

    protected override void Start()
    {
        base.Start();
        AddAllOccupancyNodes();
    }

    protected override void PerformContinuousDynamics()
    {
        if(timer>=lifeTime) Explode();
        if (rb)
        {
            rb.velocity = speed * transform.forward;
            currentPosition = rb.position;
            hasPositionChanged = true;
        }

        timer += Time.fixedDeltaTime;
    }

    private void Explode()
    {
        SubtractAllOccupancyNodes();
        Destroy(gameObject);
    }

    private void OnCollisionEnter(Collision collision)
    {
        Explode();
    }
}