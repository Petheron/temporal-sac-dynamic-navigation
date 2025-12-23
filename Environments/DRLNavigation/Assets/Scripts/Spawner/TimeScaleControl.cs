
using System;
using Unity.MLAgents;
using UnityEngine;

public static class TimeScaleControl
{
    private static float cachedTimeScale;
    
    public static Action OnTimeScaleSet;

    private static bool IsCurrentlyPaused => !(Time.timeScale > 0f);

    /// <summary>
    /// Allows for pausing/unpausing of timescale and Academy stepping.
    /// Called with True before loading/generating new level and false after agents are spawned/reset.
    /// </summary>
    /// <param name="enablePause"></param>
    public static void TogglePauseState(bool enablePause)
    {
        // Exit if already in desired pause state 
        if (enablePause == IsCurrentlyPaused) return;
        
        if (enablePause)
        {
            cachedTimeScale = Time.timeScale;
            SetTimeScale(0f);
            Academy.Instance.AutomaticSteppingEnabled = false;
        }
        else
        {
            SetTimeScale(cachedTimeScale);
            cachedTimeScale = 0f;
            Academy.Instance.AutomaticSteppingEnabled = true;
        }
        // Debug.Log($"Pause {enablePause}, Cached: {cachedTimeScale}, TimeScale: {Time.timeScale}, Stepping: {Academy.Instance.AutomaticSteppingEnabled}");
    }

    /// <summary>
    /// Set Time.timeScale to desired value whilst ensuring non-negative values.
    /// </summary>
    /// <param name="newTimeScale"></param>
    public static void SetTimeScale(float newTimeScale = 1f)
    {
        Time.timeScale = Mathf.Max(0f, newTimeScale);
        OnTimeScaleSet.Invoke();
    }
}
