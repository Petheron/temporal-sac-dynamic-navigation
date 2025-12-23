
using System.Collections.Generic;
using UnityEngine;

public static class AgentUtility
{
    public static List<float> QuaternionToList(Quaternion quaternion)
    {
        return new List<float>{quaternion.x, quaternion.y, quaternion.z, quaternion.w};
    }
    public static List<float> VectorToList(Vector3 vector)
    {
        return new List<float>{vector.x, vector.y, vector.z};
    }

    public static List<float> GetNormalizedVectorObservation(Vector3 vectorToNormalize, float normalizationValue, VectorNormalizationType normalizationType=VectorNormalizationType.Componentwise)
    {
        // prevent division by zero or negative number.
        normalizationValue = Mathf.Max(normalizationValue, Mathf.Epsilon);

        Vector3 vector = Vector3.zero;
        float magnitude = 0f;

        switch (normalizationType)
        {
            case VectorNormalizationType.Componentwise:
                NormalizeObservationVectorComponentwise(vectorToNormalize, normalizationValue, ref vector, ref magnitude);
                break;
            case VectorNormalizationType.DirectionMagnitude:
                NormalizeObservationVectorDirectionMagnitude(vectorToNormalize, normalizationValue, ref vector, ref magnitude);
                break;
        }

        List<float> output = VectorToList(vector);
        output.Add(magnitude);

        return output;
    }
    
    /// <summary>
    /// Normalizes each component of the vector by the normalization distance and the magnitude.
    /// Inspired by the normalization guidelines for MLAgents in https://github.com/Unity-Technologies/ml-agents/blob/release-21-branch/docs/Learning-Environment-Design-Agents.md#vector-observation-summary--best-practices
    /// </summary>
    private static void NormalizeObservationVectorComponentwise(Vector3 vectorToNormalize, float normalizationValue, ref Vector3 vector, ref float magnitude)
    {
        vector = vectorToNormalize / normalizationValue;
        magnitude = vectorToNormalize.magnitude / normalizationValue;
    }
    
    /// <summary>
    /// Uses traditional direction normalization and magnitude calculations.
    /// Inspired by agent observation in https://github.com/ubisoft/DirectBehaviorSpecification/blob/main/arenaEnv/arenaEnvUnityProject/Assets/Scripts/ArenaAgent.cs
    /// </summary>
    /// <param name="vectorToNormalize"></param>
    /// <param name="normalizationValue"></param>
    /// <param name="vector"></param>
    /// <param name="magnitude"></param>
    private static void NormalizeObservationVectorDirectionMagnitude(Vector3 vectorToNormalize, float normalizationValue, ref Vector3 vector, ref float magnitude)
    {
        vector = vectorToNormalize.normalized;
        magnitude = vectorToNormalize.magnitude / normalizationValue;
    }
    
    
    
    
    
}
