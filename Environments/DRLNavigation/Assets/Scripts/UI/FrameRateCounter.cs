using UnityEngine;
using TMPro;

public class FrameRateCounter : MonoBehaviour
{
    private enum DisplayMode {FPS, MS}
    
    [SerializeField] private TextMeshProUGUI display;
    [SerializeField] private DisplayMode displayMode = DisplayMode.FPS;
    [SerializeField, Range(0.1f, 2f)] private float sampleDuration = 1f;

    private int frames;
    private float duration, bestDuration = float.MaxValue, worstDuration;

    private void Update()
    {
        float frameDuration = Time.unscaledDeltaTime;
        frames += 1;
        duration += frameDuration;

        if (frameDuration < bestDuration)
        {
            bestDuration = frameDuration;
        }

        if (frameDuration > worstDuration)
        {
            worstDuration = frameDuration;
        }
        
        if (duration >= sampleDuration)
        {
            if (displayMode == DisplayMode.FPS)
            {
                display.SetText("FPS\n{0:0}", 
                    frames/duration
                );
            }
            else
            {
                display.SetText("MS\n{0:1}", 
                    1000f * duration/frames
                );
            }
            
            frames = 0;
            duration = 0f;
            bestDuration = float.MaxValue;
            worstDuration = 0f;
        }
    }
}
