using System;
using Unity.AI.Navigation;
using UnityEngine;

public class MovingPlatform : Dynamic
{
    public bool useInitialPositionAsStartPosition = true;
    public Vector3 displacementVector = Vector3.right;
    
    public float travelTime = 10f;
    private float timer;
    private bool reverse;
    private readonly AnimationCurve curve = AnimationCurve.EaseInOut(0f,0f,1f,1f);

    public bool preventSpawnPlacement = true;
    private NavMeshModifierVolume navMeshModifierVolume;
    
    private Vector3 StartPosition => useInitialPositionAsStartPosition? defaultPosition: defaultPosition-displacementVector/2f;
    private Vector3 EndPosition => useInitialPositionAsStartPosition? defaultPosition+displacementVector: defaultPosition+displacementVector/2f;

    protected override void Awake()
    {
        navMeshModifierVolume = GetComponent<NavMeshModifierVolume>();
        if (navMeshModifierVolume && preventSpawnPlacement)
        {
            BoxCollider box = GetComponent<BoxCollider>();
            if(box) navMeshModifierVolume.size = box.size;
            navMeshModifierVolume.enabled = preventSpawnPlacement;
        }
        base.Awake();
        
    }

    protected override void PerformContinuousDynamics()
    {
        if (Math.Abs(timer - travelTime) < Mathf.Epsilon) reverse = true;
        else if (timer == 0f) reverse = false;

        float deltaTime = reverse ? -Time.fixedDeltaTime : Time.fixedDeltaTime;
        timer += deltaTime;
        timer = Mathf.Clamp(timer, 0f, travelTime);
        float t = timer / travelTime;
        
        t = curve.Evaluate(t);
        
        Vector3 position = Vector3.Lerp(StartPosition, EndPosition, t);
        if (rb)
        {
            rb.MovePosition(position);
        }
        else
        {
            transform.position = position;
        }
        currentPosition = position;
        
        hasPositionChanged = true;
    }

    public override void Reset()
    {
        base.Reset();
        timer = 0f;
    }

    protected override void OnDrawGizmosSelected()
    {
        base.OnDrawGizmosSelected();
        Gizmos.color = Color.red;
        Gizmos.DrawWireCube(StartPosition,0.1f*Vector3.one);
        Gizmos.DrawWireCube(EndPosition,0.1f*Vector3.one);
        Gizmos.DrawLine(StartPosition,EndPosition);
    }
}
