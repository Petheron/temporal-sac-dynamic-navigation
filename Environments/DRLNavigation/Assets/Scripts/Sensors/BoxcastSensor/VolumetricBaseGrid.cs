using System;
using System.Collections.Generic;
using Unity.VisualScripting.FullSerializer;
using UnityEngine;

// Voxel Grid as used for basic occupancy grid map.
public class VolumetricBaseGrid:BaseGrid
{
    private int baseGridSize = 8;
    private float baseCellSize = 1f;
    private int[] LODs;
    private Vector3 rootPosition;

    private float cellSizePercentageBuffer;
    private int baseNumCells;

    private Voxel[][] arrays;


    public VolumetricBaseGrid(int gridSize, float cellSize, int levelsOfDetail = 1, int numCategories=1, Vector3 rootPos = new Vector3(), int[] values=null)
    {
        baseGridSize = gridSize;
        baseCellSize = cellSize;
        LODs = CalculateLODs(levelsOfDetail);
        arrays = new Voxel[levelsOfDetail][];
        baseNumCells = NumCells();
        // Move rootposition by a half grid. By default centre of grid is at (0,0,0).
        rootPosition = rootPos - (0.5f*baseGridSize + 0.5f)*baseCellSize*Vector3.one;
        
        for (int i = 0; i < LODs.Length; i++)
        {
            int LOD = LODs[i];
            int n = NumCells(LOD);
            float size = CellSize(LOD);
            arrays[i] = new Voxel[n];
            
            for (int j = 0; j < n; j++)
            {
                Vector3 centre = GetLocalPosition(i, LOD);
                arrays[i][j] = new Voxel(centre,size, new VoxelData(numCategories, values));
            }
        }
    }

    public void Initialize(ref Voxel[] fineArray)
    {
        Debug.Assert(fineArray.Length==baseNumCells);
        for (int i = 0; i < LODs.Length; i++)
        {
            arrays[i] = GetLODGrid(ref fineArray, i);
        }
    }

    /// <summary>
    /// LODs stored (1,2,4,...).
    /// </summary>
    /// <param name="levelsOfDetail">How many levels to store.</param>
    /// <returns></returns>
    private int[] CalculateLODs(int levelsOfDetail=1)
    {
        List<int> list = new List<int>();
        for (int i = 0; i < levelsOfDetail; i++)
        {
            list.Add((int)Mathf.Pow(2f,i));
        }
        return list.ToArray();
    }

    private Voxel[] GetLODGrid(ref Voxel[] fineArray, int LODIndex = 0)
    {
        int LOD = GetLOD(LODIndex);
        if (LOD == 1) return fineArray;

        List<Voxel> list = new List<Voxel>();
        
        for (int y = 0; y < baseGridSize; y+=LOD)
        {
            for (int z = 0; z < baseGridSize; z+=LOD)
            {
                for (int x = 0; x < baseGridSize; x+=LOD)
                {
                    Voxel v = ConstructCoarseVoxel(ref fineArray, x, y, z, LODIndex);
                    if(v.IsNull()) continue;
                    list.Add(v);
                }
            }
        }
        return list.ToArray();
    }
    
    private Voxel ConstructCoarseVoxel(ref Voxel[] fineArray, int x, int y, int z, int LODIndex=0)
    {
        int LOD = GetLOD(LODIndex);
        int initIndex = GetGridIndex(x, y, z);
        if (initIndex < 0) return Voxel.NullVoxel();
        Voxel voxel = Voxel.EmptyVoxel(fineArray[initIndex].Data);

        for (int j = y; j < y+LOD; j++)
        {
            for (int k = z; k < z+LOD; k++)
            {
                for (int i = x; i < x+LOD; i++)
                {
                    int idx = GetGridIndex(i, j, k, 0);
                    if (idx == initIndex) continue;
                    
                    Voxel v = IsValidGridIndex(idx) ? fineArray[idx] : Voxel.NullVoxel();
                    voxel.CombineData(v.Data);
                }
            }
        }

        int index = GetGridIndex(x / LOD, y / LOD, z / LOD, LODIndex);
        voxel.Centre = GetLocalPosition(index, LODIndex);
        voxel.Size = CellSize(LODIndex);
        
        return voxel;
    }

    
    /// <summary>
    /// Returns idx = x + Xz + XZy.
    /// </summary>
    /// <returns>Grid index for a given LOD.</returns>
    public int GetGridIndex(int x, int y, int z, int LODIndex=0)
    {
        if (!IsValidDimIndex(x,LODIndex)) return -1;
        if (!IsValidDimIndex(y,LODIndex)) return -1;
        if (!IsValidDimIndex(z,LODIndex)) return -1;
        int g = GridSize(LODIndex);
        int i =  x + g * z + g * g * y;
        return IsValidGridIndex(i,LODIndex) ? i : -1;
    }

    private Vector3Int GetDimIndexForGridSize(int idx, int X, int Y ,int Z)
    {
        if (idx < 0 || idx >= X*Y*Z) return -Vector3Int.one;
        int i = idx;
        int y = i / (X * Z);
        i %= X * Z;
        int z = i / X;
        int x = i % X;
        return new Vector3Int(x, y, z);
    }

    public Vector3Int GetDimIndex(int index, int LODIndex = 0)
    {
        if (!IsValidGridIndex(index, LODIndex)) return -Vector3Int.one;
        int g = GridSize(LODIndex);
        return GetDimIndexForGridSize(index, g, g, g);
    }

    public int GetGridIndexFromGlobalPosition(Vector3 position, int LODIndex=0)
    {
        Vector3Int dims = GetDimsFromGlobalPosition(position,LODIndex);
        int idx = GetGridIndex(dims.x, dims.y, dims.z, LODIndex);
        return IsValidGridIndex(idx,LODIndex)? idx : -1;
    }

    public Vector3Int GetDimsFromGlobalPosition(Vector3 position, int LODIndex = 0)
    {
        position -= rootPosition;
        position /= CellSize(LODIndex);
        Vector3Int intDim = new Vector3Int();
        for (int i = 0; i < 3; i++)
        {
            intDim[i] = Mathf.FloorToInt(position[i]);
        }   
        return intDim;
    }

    public Vector3 GetGlobalPositionFromIndex(int idx)
    {
        return baseCellSize*(Vector3)GetDimIndexForGridSize(idx,baseGridSize,baseGridSize,baseGridSize) + rootPosition + 0.5f*baseCellSize*Vector3.one;
    }

    public Vector3 GetCellGlobalPosition(int idx, int LODIndex=0)
    {
        Voxel v = GetVoxel(idx,LODIndex);
        if(v.IsNull()) return GetLocalPosition(idx,LODIndex) + rootPosition;
        return rootPosition + v.Centre;
    }

    public Vector3 GetLocalPosition(int idx, int LODIndex=0)
    {
        if (!IsValidGridIndex(idx, LODIndex)) return Vector3.negativeInfinity;
        Vector3 dims = GetDimIndex(idx, LODIndex);
        float size = CellSize(LODIndex);
        return dims * size + size / 2f * Vector3.one;
    }

    public void Sample(Vector3 globalPosition, Vector3Int sampleSize, out int[] values, out int[] indices, int initialLODIndex=0, int numChannels=1)
    {
        sampleSize = GetOddGrid(sampleSize);
        int n = sampleSize.x * sampleSize.y * sampleSize.z;

        values = new int[n*numChannels];
        indices = new int[n*numChannels];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < numChannels; j++)
            {
                int channelLODIndex = initialLODIndex + j;
                Debug.Assert(IsValidLOD(channelLODIndex));
                int idx = GetGridIndexFromGlobalPosition(globalPosition, channelLODIndex);
                int sampleIdx = GetGridIndexForSampleIndex(idx, i, sampleSize, channelLODIndex);
                bool isValid = IsValidGridIndex(sampleIdx, channelLODIndex);
                Voxel v = GetVoxel(sampleIdx, channelLODIndex);
                indices[j+numChannels*i] = isValid ? sampleIdx : -1;
                values[j+numChannels*i] = !v.IsNull() ? v.Value() : -1;
            }
            
        }
    }

    private int GetGridIndexForSampleIndex(int centreGridIndex, int sampleIndex, Vector3Int sampleSize, int LODIndex =0)
    {
        if (!IsValidGridIndex(centreGridIndex, LODIndex)) return -1;
        
        Vector3Int sampleCentreDim = new Vector3Int((sampleSize.x - 1) / 2, (sampleSize.y - 1) / 2, (sampleSize.z - 1) / 2);
        Vector3Int sampleIndexDim = GetDimIndexForGridSize(sampleIndex, sampleSize.x, sampleSize.y, sampleSize.z);
        Vector3Int intOffset = sampleIndexDim - sampleCentreDim;

        Vector3Int centreDimInGrid = GetDimIndex(centreGridIndex, LODIndex);
        Vector3Int sampleDimInGrid = centreDimInGrid + intOffset;

        return GetGridIndex(sampleDimInGrid.x, sampleDimInGrid.y, sampleDimInGrid.z, LODIndex);
    }

    public Voxel GetVoxel(int index, int LODIndex = 0)
    {
        if (!IsValidGridIndex(index, LODIndex)) return Voxel.NullVoxel();
        return arrays[LODIndex][index];
    }

    private void SetVoxel(Voxel v, int index, int LODIndex = 0)
    {
        if (!IsValidGridIndex(index, LODIndex)) return;
        arrays[LODIndex][index] = v;
    }

    public void AddNewDynamicCount(int[] idxs, int tagValue)
    {
        if (tagValue <= 0) return;
        
        // loop through each LOD
        for (int i = 0; i < idxs.Length; i++)
        {
            int idx = idxs[i];

            // Increment voxel being considered
            Voxel voxel = GetVoxel(idx, i);
            if (!voxel.IsNull())
            {
                voxel.Data.IncrementCategoryCount(tagValue);
                SetVoxel(voxel, idx, i);
            }
        }
    }
    
    public void SubtractDespawningDynamicCount(int[] idxs, int tagValue)
    {
        if (tagValue <= 0) return;
        
        // loop through each LOD
        for (int i = 0; i < idxs.Length; i++)
        {
            // if(i>0) continue;
            int idx = idxs[i];

            // Decrement voxel being considered
            Voxel voxel = GetVoxel(idx, i);
            if (!voxel.IsNull())
            {
                voxel.Data.DecrementCategoryCount(tagValue);
                SetVoxel(voxel, idx, i);
            }
        }
    }

    public void UpdateDynamicCount(int[] oldIdxs, int[] newIdxs, int tagValue)
    {
        if (tagValue <= 0) return;

        // loop through each LOD
        for (int i = 0; i < oldIdxs.Length; i++)
        {
            // if(i>0) continue;
            int oldIdx = oldIdxs[i];
            int newIdx = newIdxs[i];
            
            // // ignore unchanged index
            if(oldIdx==newIdx) continue;
            
            // Decrement voxel being exited
            Voxel oldVoxel = GetVoxel(oldIdx, i);
            if (!oldVoxel.IsNull())
            {
                oldVoxel.Data.DecrementCategoryCount(tagValue);
                SetVoxel(oldVoxel, oldIdx, i);
            }

            // Increment voxel being entered
            Voxel newVoxel = GetVoxel(newIdx, i);
            if (!newVoxel.IsNull())
            {
                newVoxel.Data.IncrementCategoryCount(tagValue);
                SetVoxel(newVoxel, newIdx, i);
            }
        }
    }

    private int GetLOD(int LODIndex = 0)
    {
        LODIndex = Mathf.Clamp(LODIndex, 0, LODs.Length - 1);
        return LODs[LODIndex];
    }

    private bool IsValidDimIndex(int i, int LODIndex = 0)
    {
        if (!IsValidLOD(LODIndex)) return false;
        if (i < 0) return false;
        return i < GridSize(LODIndex);
    }
    
    private bool IsValidGridIndex(int index, int LODIndex = 0)
    {
        if (index < 0) return false;
        if (!IsValidLOD(LODIndex)) return false;
        return index < NumCells(LODIndex);
    }

    private bool IsValidLOD(int LODIndex)
    {
        return LODIndex >= 0 && LODIndex < LODs.Length;
    }
    
    private int GridSize(int LODIndex = 0)
    {
        if (!IsValidLOD(LODIndex)) return 0;
        return baseGridSize / GetLOD(LODIndex);
    }
    public float CellSize(int LODIndex = 0)
    {
        if (!IsValidLOD(LODIndex)) return 0;
        return baseCellSize * GetLOD(LODIndex);
    }

    public int NumCells(int LODIndex = 0)
    {
        int g = GridSize(LODIndex);
        return g * g * g;
    }

    public int GetBaseNumCells()
    {
        return baseNumCells;
    }
    


    private float GetSize()
    {
        return baseGridSize * baseCellSize;
    }

}
