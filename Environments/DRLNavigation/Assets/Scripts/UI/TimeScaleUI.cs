using TMPro;
using UnityEngine;

public class TimeScaleUI : MonoBehaviour
{
    [SerializeField] private int defaultTimescale = 1;
    public TMP_Text timeScaleText;

    private void Awake()
    {
        TimeScaleControl.OnTimeScaleSet += SetTimeScaleText;
        
#if UNITY_EDITOR
        SetTimeScale(defaultTimescale);
#endif
        
    }

    private void Update()
    {
#if UNITY_EDITOR
        if (Input.GetButtonDown("TimeScale"))
        {
            SetTimeScale(Time.timeScale + Input.GetAxis("TimeScale"));
        }
#endif
    }

    private void SetTimeScaleText()
    {
        if (!timeScaleText) return;
        timeScaleText.text = $"Time Scale: {Time.timeScale}";
    }

    private void SetTimeScale(float newTimeScale)
    {
        if (newTimeScale <= 0) return;
        TimeScaleControl.SetTimeScale(newTimeScale);
    }
}
