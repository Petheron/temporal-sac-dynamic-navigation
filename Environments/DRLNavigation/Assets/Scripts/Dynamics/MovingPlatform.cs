using System;
using Unity.AI.Navigation;
using UnityEngine;
using Unity.MLAgents;

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
    
    private EnvironmentParameters environmentParameters;

    // Whether each episode starts the platform at a random point in its cycle.
    private bool randomisePhase;
    
    private static bool hasLoggedPlatformTiming;
    
    private Vector3 StartPosition => useInitialPositionAsStartPosition? defaultPosition: defaultPosition-displacementVector/2f;
    private Vector3 EndPosition => useInitialPositionAsStartPosition? defaultPosition+displacementVector: defaultPosition+displacementVector/2f;

    protected override void Awake()
    {
        InitializeEnvironmentParameters();

        navMeshModifierVolume = GetComponent<NavMeshModifierVolume>();
        if (navMeshModifierVolume && preventSpawnPlacement)
        {
            BoxCollider box = GetComponent<BoxCollider>();
            if(box) navMeshModifierVolume.size = box.size;
            navMeshModifierVolume.enabled = preventSpawnPlacement;
        }
        base.Awake();

    }

    private void InitializeEnvironmentParameters()
    {
        environmentParameters = Academy.Instance.EnvironmentParameters;

        // Defaults to inspector value.
        travelTime = environmentParameters.GetWithDefault("platform_travel_time", travelTime);

        // Phase randomisation stays off unless a config asks for it.
        randomisePhase = environmentParameters.GetWithDefault("randomise_platform_phase", 0f) > 0f;

        if (hasLoggedPlatformTiming) return;
        hasLoggedPlatformTiming = true;
        
        // Logged once so the traverse timing is recorded in the run log.
        int physicsStepsPerTraverse = Mathf.RoundToInt(travelTime / Time.fixedDeltaTime);
        Debug.Log($"[PLATFORM] travelTime={travelTime:F3} s " +
                  $"fixedDeltaTime={Time.fixedDeltaTime:F4} s " +
                  $"physicsStepsPerTraverse={physicsStepsPerTraverse} " +
                  $"randomisePhase={randomisePhase}");
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
        
        timer = randomisePhase ? UnityEngine.Random.Range(0f, travelTime) : 0f;

        // PerformContinuousDynamics only clears reverse when the timer is exactly zero; a randomised start never reaches that branch, so the direction is set explicitly here.
        reverse = randomisePhase && UnityEngine.Random.value < 0.5f;
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
