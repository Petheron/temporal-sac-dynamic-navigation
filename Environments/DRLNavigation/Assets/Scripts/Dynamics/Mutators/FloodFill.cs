using System;
using System.Collections;
using System.Collections.Generic;
using LevelGeneration;
using UnityEngine;

public class FloodFill : MonoBehaviour
{
    [SerializeField] private MutatorType mutatorType = MutatorType.Water;
    [SerializeField] private float maxHeight;
    [SerializeField] private OccupancyMap occupancyMap;
    [SerializeField] private LevelGenerator levelGenerator;

    [SerializeField] private Transform seedTransform;
    private byte[] visited;
    private int currentSize;

    private int width;
    private int height;
    private int depth;

    [SerializeField, ReadOnly] private int maxHeightDimIndex;

    public bool test;

    private static Vector3Int[] neighbourOffset = new[]
    {
        new Vector3Int(1, 0, 0),
        new Vector3Int(-1, 0, 0),
        new Vector3Int(0, 1, 0),
        new Vector3Int(0, -1, 0),
        new Vector3Int(0, 0, 1),
        new Vector3Int(0, 0, -1),
    };

    private void Start()
    {
        // Fill();
    }

    private void FixedUpdate()
    {
        if (test)
        {
            Fill();
            test = false;
        }
    }

    private void Fill()
    {
        FindOccupancyMap();
        if(!occupancyMap) return;

        if (!seedTransform) seedTransform = transform;
        Vector3 seedPosition = seedTransform.position;
        if (maxHeight < seedPosition.y)
        {
            maxHeight = seedPosition.y;
        }

        width = occupancyMap.occupancyGridProperties.gridSize;
        height = occupancyMap.occupancyGridProperties.gridSize;
        depth = occupancyMap.occupancyGridProperties.gridSize;

        visited = new byte[width*height*depth];

        Vector3Int startPos = occupancyMap.GetDimIndexFromGlobalPosition(seedPosition, 0);
        Vector3 maxHeightPos = new Vector3(seedPosition.x, maxHeight, seedPosition.z);
        maxHeightDimIndex = occupancyMap.GetDimIndexFromGlobalPosition(maxHeightPos, 0).y;
        
        IterativeFill(startPos.x,startPos.y,startPos.z);
        
        Visualize();
    }

    private void IterativeFill(int i, int j, int k)
    {
        var stack = new Stack<Vector3Int>();
        int idx = occupancyMap.GetGridIndex(i, j, k,0);
        if (idx < 0) return;
        stack.Push(new Vector3Int(i,j,k));
        while (stack.Count>0)
        {
            var current = stack.Pop();
            var currentIdx = occupancyMap.GetGridIndex(current.x, current.y, current.z,0);
            
            if(visited[currentIdx]==1) continue;
            
            var cellPosition = occupancyMap.GetCellGlobalPosition(currentIdx, 0);
            occupancyMap.AddNewDynamicOccupancyIndex(occupancyMap.GetAllIndicesFromGlobalPosition(cellPosition),VisionTag.Solid,mutatorType);
            visited[currentIdx] = 1;

            foreach (var offset in neighbourOffset)
            {
                var newDimIdx = current + offset;
                if (newDimIdx.y > maxHeightDimIndex) continue;
                var newIdx = occupancyMap.GetGridIndex(newDimIdx.x, newDimIdx.y, newDimIdx.z);
                if(newIdx<0) continue;
                if (visited[newIdx] == 1) continue;
                if (occupancyMap.GetValue(newIdx, 0) != 0) continue;
                stack.Push(newDimIdx);
                
            }
        }
    }

    private void Visualize()
    {
        // loop through visited to see where water is.
        // take logic from level generator / chunk and use it here to visualize.
        // alternatively, update chunk logic to work with 1D array, and work with provided global grid (subdivide a given grid and provide to relevant chunks).
        if (levelGenerator)
        {
            levelGenerator.cellSize = occupancyMap.CellSize(0);
            levelGenerator.Generate(visited, new Vector3Int(width,height,depth));
        }
    }

    private void FindOccupancyMap()
    {
        if (!occupancyMap)
        {
            occupancyMap = FindObjectOfType<OccupancyMap>();
        };
    }
}
