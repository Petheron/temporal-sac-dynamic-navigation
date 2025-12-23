using System.Collections;
using System;
using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class DepthmapPerception
{
    private int height;
    private int width;
    private float maxAngle;
    private int numChannels;
    private int arrayLength;
    private VisibleTags visibleTags;
    /// <summary>
    /// The furthest distance travelled by rays.
    /// </summary>
    private float maxDistance;
    private LayerMask groundedMask;
    private Transform t;

    public float[] debugRayLengths;
    public Vector3[] debugRayDirections;

    public event Action<float[]> DepthmapSample;
    public event Action<float[]> DebugDepthmapSample;
    
    public DepthmapPerception
    (
        int height,
        int width,
        float maxAngle,
        float maxDistance,
        LayerMask groundedMask,
        VisibleTags visibleTags,
        int numChannels,
        Transform transform
        )
    {
        this.height = height;
        this.width = width;
        this.maxAngle = maxAngle;
        this.maxDistance = maxDistance;
        this.groundedMask = groundedMask;
        this.visibleTags = visibleTags;
        this.numChannels = numChannels;
        t = transform;

        arrayLength = height * width * numChannels;

        debugRayLengths = new float[arrayLength];
        debugRayDirections = new Vector3[arrayLength];
    }
    
    public void Perceive()
    {
        float[] sample = CastRays();
        DepthmapSample.Invoke(sample);
        DebugDepthmapSample.Invoke(sample);
    }

    private int GetPosIndex(int h, int w, int c)
    {
        if (h < 0 || h >= height) return -1;
        if (w < 0 || w >= width) return -1;
        if (c < 0 || c >= numChannels) return -1;
        return c + numChannels * w + width * numChannels * h;
    }

    /// <summary>
    /// Cast rays in agent facing direction in a (width, height) grid. 
    /// </summary>
    /// <returns>An array of normalized distances between 0 and 1 indexed at (h*W + w).</returns>
    private float[] CastRays()
    {
        float[] distanceArray = new float[arrayLength];
        
        // Rays are cast from sensor position. Forms a square of size (2*squareHalfSize, 2*squareHalfSize) at distance squareDepth.
        var rayOrigin = t.position;
        var squareHalfSize = maxDistance * Mathf.Sin(Mathf.Deg2Rad * maxAngle);
        var squareDepth = maxDistance * Mathf.Cos(Mathf.Deg2Rad * maxAngle);
        // Calculate size of cells of grid.
        var cellWidth = 2 * squareHalfSize / width;
        var cellHeight = 2 * squareHalfSize / height;
        var up = t.up;
        var right = t.right;
        // Determine initial position to begin loop. Start at top-left as Texture2D has (0,0) in top-left.
        // Subtract half-subwidths to ensure position in cell centre.
        var squareTopLeft = rayOrigin + t.forward * squareDepth + (squareHalfSize-0.5f*cellHeight)*up - (squareHalfSize-0.5f*cellWidth)*right;

        // For each cell
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                // Cast ray in direction of cell for distance maxDistance.
                var cellPosition = squareTopLeft + right * (cellWidth * w) - up * (cellHeight * h);
                var vectorToCell = cellPosition - rayOrigin;
                var directionToCell = vectorToCell.normalized;
                var rayLength = vectorToCell.magnitude;
                // var rayLength = m_MaxDistance;

                for (int c = 0; c < numChannels; c++)
                {
                    int i = GetPosIndex(h, w, c);
                    if (i<0) continue;
                    debugRayLengths[i] = rayLength;
                    debugRayDirections[i] = directionToCell;
                }

                // If object hit, cache distance else use rayLength
                var distance = rayLength;
                var channelIndex = -1;
                if (Physics.Raycast(rayOrigin, directionToCell, out var hit, rayLength, groundedMask,
                        QueryTriggerInteraction.Collide))
                {
                    distance = hit.distance;
                    channelIndex = visibleTags.IndexFromString(hit.collider.tag);
                }
                int idx = GetPosIndex(h, w, channelIndex);
                if (idx<0) continue;
                // Normalize value such that 0 if nothing hit or 1 if point blank collision.
                distanceArray[idx] = (rayLength - distance) / rayLength;
            }
        }
        return distanceArray;
    }

    public void RegisterSensor(DepthmapSensor sensor)
    {
        DepthmapSample += sensor.ProcessSampleArray;
    }

    public void RegisterDebugSensor(DepthmapSensor debugSensor)
    {
        DebugDepthmapSample += debugSensor.ProcessSampleArray;
    }
}
