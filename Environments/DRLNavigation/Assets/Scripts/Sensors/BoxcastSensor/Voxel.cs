using Unity.VisualScripting.FullSerializer;
using UnityEngine;

public struct Voxel
{
    public Vector3 Centre;
    public float Size;
    public VoxelData Data;

    public Voxel(Vector3 c, float s, VoxelData d)
    {
        Centre = c;
        Size = s;
        Data = d;
    }

    public static Voxel NullVoxel()
    {
        return new Voxel(Vector3.zero, 0f, new VoxelData(0, null));
    }

    public static Voxel EmptyVoxel(VoxelData data)
    {
        return new Voxel(Vector3.zero, 0f, new VoxelData(data.GetNumCategories(), data.GetValues()));
    }

    public int Value()
    {
        return Data.Value;
    }

    public int[] Counts()
    {
        return Data.GetCounts();
    }

    public void CombineData(VoxelData otherData)
    {
        if (IsNull()) return;
        if (otherData.IsNull()) return;
        if (otherData.IsEmpty()) return;
        Data.Combine(otherData);
    }

    public bool IsEmpty()
    {
        return Data.IsEmpty();
    }

    public bool IsNull()
    {
        return Data.IsNull();
    }
}

public struct VoxelData
{
    public int Value;
    private int numCategories;
    private int[] counts;
    private int[] values;
    private int totalCount;

    public VoxelData(int numCats, int[] vals = null)
    {
        Value = 0;
        numCategories = Mathf.Max(numCats,1);
        counts = new int[numCategories];
        if (vals == null)
        {
            values = new int[numCategories];
            for (int i = 0; i < numCategories; i++)
            {
                values[i] = i + 1;
            }
        }
        else
        {
            values = vals;
        }
        totalCount = 0;
    }

    public void SetCategoryValue(int newCountValue, int catIndex)
    {
        SetCount(newCountValue, catIndex);
        UpdateValue();
    }

    private void SetCount(int newCountValue, int catIndex)
    {
        Debug.Assert(IsValidCategoryIndex(catIndex));
        int oldCount = counts[catIndex];
        if (newCountValue < 0) newCountValue = 0;
        counts[catIndex] = newCountValue;
        totalCount += newCountValue - oldCount;
    }

    public void DecrementCategoryCount(int value)
    {
        int catIndex = ValueToIndex(value);
        if (!IsValidCategoryIndex(catIndex)) return;
        int newValue = counts[catIndex] - 1;
        SetCategoryValue(newValue, catIndex);
    }
    
    public void IncrementCategoryCount(int value)
    {
        int catIndex = ValueToIndex(value);
        if (!IsValidCategoryIndex(catIndex)) return;
        int newValue = counts[catIndex] + 1;
        SetCategoryValue(newValue, catIndex);
    }

    private void UpdateValue()
    {
        int maxIndex = -1;
        for (int i = counts.Length-1; i >= 0; i--)
        {
            if (counts[i] <= 0) continue;
            maxIndex = i;
            break;
        }

        if (maxIndex < 0)
        {
            Value = 0;
            return;
        }
        // Value is equal to maxIndex + 1 if counter > 0, else 0
        // Value = counts[maxIndex]>0?maxIndex+1:0;
        Value = counts[maxIndex] > 0 ? values[maxIndex] : 0;
    }

    public void Combine(VoxelData other)
    {
        int[] otherCounts = other.GetCounts();
        for (int i = 0; i < numCategories; i++)
        {
            int count = otherCounts[i] + counts[i];
            SetCount(count,i);
        }
        UpdateValue();
    }

    public int[] GetCounts()
    {
        return counts;
    }

    public int[] GetValues()
    {
        return values;
    }

    public int TotalCount()
    {
        return totalCount;
    }

    public void SetCounts(int[] newCounts)
    {
        for (int i = 0; i < counts.Length; i++)
        {
            if (!IsValidCategoryIndex(i)) return;
            SetCount(newCounts[i],i);
        }
        UpdateValue();
    }

    public bool IsEmpty()
    {
        return totalCount <= 0;
    }

    public bool IsNull()
    {
        return numCategories <= 0;
    }

    private int ValueToIndex(int value)
    {
        int idx = -1;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] == value)
            {
                idx = i;
                break;
            }
        }
        return idx;
    }

    private bool IsValidCategoryIndex(int index)
    {
        return index >= 0 && index < counts.Length;
    }

    public int GetNumCategories()
    {
        return numCategories;
    }
}
