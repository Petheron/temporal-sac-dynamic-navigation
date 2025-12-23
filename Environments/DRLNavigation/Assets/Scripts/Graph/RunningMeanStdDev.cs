using System;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.InputSystem.Controls;
using Random = UnityEngine.Random;

/// <summary>
/// Calculate the running mean and variance for a given input.
/// Uses Welford's Online algorithm as less prone to catastrophic cancellation.
/// </summary>

[JsonObjectAttribute]
public class RunningMeanStdDev
{
    [JsonPropertyAttribute] private float count;
    [JsonPropertyAttribute] private float sumOfSquares;
    [JsonPropertyAttribute] private float mean;
    

    public RunningMeanStdDev()
    {
        Reset();
    }

    // public RunningMeanStdDev(float startingValue)
    // {
    //     Reset();
    //     Update(startingValue);
    // }

    public void Reset()
    {
        mean = 0f;
        count = 0;
        sumOfSquares = 0f;
    }

    public void Update(float newValue)
    {
        count++;
        float delta = newValue - mean;
        mean += delta / count;
        float delta2 = newValue - mean;
        sumOfSquares += delta * delta2;
        
        // Debug.Log($"new weight: {newValue}, updated mean: {mean}, updated std dev: {GetStdDev()}");
    }

    public float GetMean()
    {
        return mean;
    }

    public float GetVariance()
    {
        if (count<=0)
        {
            return 0f;
        }
        return sumOfSquares/count;
    }

    private float GetSampleVariance()
    {
        return sumOfSquares / (count - 1);
    }

    public float GetStdDev()
    {
        return Mathf.Sqrt(GetVariance());
    }

    public float SampleValue(float min=Mathf.NegativeInfinity, float max=Mathf.Infinity)
    {
        return Mathf.Clamp(SampleGaussian(GetMean(),GetStdDev()), min, max);
    }

    /// <summary>
    /// Sample perform sampling of Gaussian according to Marsaglia Polar Method.
    /// </summary>
    /// <param name="mean">Mean of Gaussian distribution</param>
    /// <param name="stdDev">Standard Deviation of Gaussian  distribution</param>
    /// <returns>Sample from desired gaussian distribution.</returns>
    private float SampleGaussian(float mean = 0f, float stdDev = 1.0f)
    {
        float u=0f, v=0f, s=0f;
        while (s is >= 1.0f or <= 0f)
        {
            u = 2.0f*Random.value - 1.0f;
            v = 2.0f*Random.value - 1.0f;
            s= u*u + v*v;
        }
        float normalSample = u*Mathf.Sqrt(-2.0f*Mathf.Log(s)/s);
        return mean + stdDev * normalSample;

    }
    
    // private float NormalOffset(float min, float max){
    //     // Samples a Normal Gaussian according to Marsaglia Polar Method. Clamps values to 3 std deviations 
    //     float u=0f, v=0f, s=0f;
    //     while(s is >= 1 or <= 0){
    //         u = Random.value*2.0f - 1.0f;
    //         v = Random.value*2.0f - 1.0f;
    //         s= u*u + v*v;
    //     }
    //     float std = u*Mathf.Sqrt(-2.0f*Mathf.Log(s)/s);
    //     float mean = (min+max)/2.0f;
    //     float sigma = (max-min)/3.0f;
    //     return Mathf.Clamp(std*sigma + mean,min,max);
    // }
    
}
