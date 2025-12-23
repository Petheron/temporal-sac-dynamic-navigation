using System;

public class ActionTimer
{
    private float currentTime;
    private readonly float maxTime;

    public event Action OnFinished;

    public bool HasStarted => currentTime < maxTime;
    public bool IsFinished => currentTime <= 0f;
    
    public ActionTimer(float time)
    {
        maxTime = time;
        Reset();
    }

    public float GetRemainingTime(bool normalized=true)
    {
        if(normalized) return currentTime / maxTime;
        return currentTime;
    }

    public void Decrement(float amount)
    {
        if (!(currentTime > 0f)) return;
        currentTime -= amount;
        if (currentTime <= 0f)
        {
            Finish();
        }
    }

    public void Start()
    {
        currentTime = maxTime;
    }

    private void Finish()
    {
        Reset();
        OnFinished?.Invoke();
    }

    public void Reset()
    {
        currentTime = 0f;
    }
}
