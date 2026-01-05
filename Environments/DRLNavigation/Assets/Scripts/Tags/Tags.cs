using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public struct VisibleData<T> where T:Enum
{
    public T Data;
    public string[] Names;
    public int[] Values;
    public int MaxValue;
    
    public VisibleData(T data) : this()
    {
        SetData(data);
    }

    private void SetData(T data)
    {
        Data = data;
        Names = GetNames();
        Values = GetValues();
        MaxValue =  Enum.GetValues(typeof(T)).Cast<int>().Max();
    }

    private string[] GetNames()
    {
        List<string> list = new List<string>();
        list.AddRange(Data.ToString().Split(", "));
        list.Remove(MutatorType.None.ToString());
        return list.ToArray();
    }

    private int[] GetValues()
    {
        List<int> values = new List<int>();
        foreach (var v in Enum.GetValues(typeof(T)).Cast<int>())
        {
            if (((int)(object)Data & v) > 0) values.Add(v);
        }
        return values.ToArray();
    }

    /// <summary>
    /// Set data according to a Base10 Integer Mask. (i.e. 10111 --> 23)
    /// </summary>
    public bool InjectBase10Mask(int mask)
    {
        if (mask < 0) return false;

        int total=0;
        int div=1;
        int n=0;
        while(div<=mask){
            int v = (mask / div) % 10;
            v *= 1<<n; //bitshift to apply power of 2^n
            if (v > MaxValue) break;
            total+=v;
            div *= 10;
            n++;
        }
        SetData((T)(total as object));
        return true;
    }
}

public class VisibleTags
{
    private VisibleData<VisionTag> visibleVisionData;
    private VisibleData<MutatorType> visibleMutatorData;

    private int visibleVisionTagCount;
    private int visibleMutatorCount;

    private string[] combinedStrings;
    private int[] combinedValues;
    
    public VisibleTags(VisionTag t, MutatorType m = MutatorType.None)
    {
        UpdateTagData(t,m);
    }

    public VisionTag GetVisionTags()
    {
        return visibleVisionData.Data;
    }

    public MutatorType GetVisibleMutatorTypes()
    {
        return visibleMutatorData.Data;
    }

    public void UpdateTagData(VisionTag t, MutatorType m = MutatorType.None)
    {
        visibleVisionData = new VisibleData<VisionTag>(t);
        visibleMutatorData = new VisibleData<MutatorType>(m);
        UpdateCombinedArrays();
    }

    public void SetDataFromBitMasks(int tagVisionMask = -1, int mutatorMask = -1)
    {
        if(visibleVisionData.InjectBase10Mask(tagVisionMask) | visibleMutatorData.InjectBase10Mask(mutatorMask)) UpdateCombinedArrays();
    }

    public string[] GetAllVisibleTags()
    {
        return combinedStrings;
    }

    public int[] GetAllVisibleValues()
    {
        return combinedValues;
    }

    public int GetValue(VisionTag visionTag, bool indexIsValue=false)
    {
        int idx = GetTagIndex(visionTag);
        return indexIsValue ? idx + 1 : combinedValues[idx];
    }
    
    public int GetValue(MutatorType mutatorType, bool indexIsValue=false)
    {
        int idx = GetMutatorIndex(mutatorType);
        int value = indexIsValue ? idx + 1: combinedValues[idx];
        return value;
    }

    public int[] GetAllVisibleTagIndices()
    {
        List<int> indices = new List<int>();
        for (int i = 0; i < combinedValues.Length; i++)
        {
            indices.Add(i+1); // offset by 1 as 0 is empty voxel.
        }
        return indices.ToArray();
    }

    private int GetTagIndex(VisionTag tag)
    {
        if (!IsVisibleTag(tag)) return -1;
        return IndexFromValue(tag);
    }

    private int GetMutatorIndex(MutatorType type)
    {
        if (!IsVisibleMutator(type)) return -1;
        return IndexFromValue(type);
    }

    public int GetVisibleTagsCount()
    {
        return combinedStrings?.Length ?? 0;
    }

    public static void AssignTag(GameObject gameObject, VisionTag tag, MutatorType type = MutatorType.None)
    {
        if(!gameObject) return;
        string name = "Untagged";
        if (type != MutatorType.None) name = type.ToString();
        else if ((int)tag > 0) name = tag.ToString();
        gameObject.tag = name;
    }

    private bool IsVisibleTag(VisionTag tag)
    {
        return visibleVisionData.Data.HasFlag(tag);
    }

    private bool IsVisibleMutator(MutatorType type)
    {
        return visibleMutatorData.Data.HasFlag(type);
    }

    private int IndexFromValue(MutatorType type)
    {
        for (int i = 0; i < visibleMutatorData.Values.Length; i++)
        {
            if ((int) type == visibleMutatorData.Values[i])
            {
                return i + visibleVisionData.Values.Length;
            }
        }
        return -1;
    }

    private int IndexFromValue(VisionTag tag)
    {
        for (int i = 0; i < visibleVisionData.Values.Length; i++)
        {
            if ((int) tag == visibleVisionData.Values[i])
            {
                return i;
            }
        }
        return -1;
    }

    public int IndexFromString(string tag)
    {
        for (int i = 0; i < combinedStrings.Length; i++)
        {
            if (tag.Equals(combinedStrings[i])) return i;
        }
        return -1;
    }

    #region Combined Region

    private void UpdateCombinedArrays()
    {
        var stringList = new List<string>();
        stringList.AddRange(visibleVisionData.Names);
        stringList.AddRange(visibleMutatorData.Names);
        combinedStrings = stringList.ToArray();
        
        var valueList = new List<int>();
        valueList.AddRange(visibleVisionData.Values);

        int[] mutatorValues = new int[visibleMutatorData.Values.Length];
        Array.Copy(visibleMutatorData.Values,mutatorValues,mutatorValues.Length);
        for (int i = 0; i < mutatorValues.Length; i++)
        {
            mutatorValues[i] *= visibleVisionData.MaxValue * 2;
        }
        valueList.AddRange(mutatorValues);
        combinedValues = valueList.ToArray();

        // for (int i = 0; i < stringList.Count; i++)
        // {
        //     Debug.Log($"{stringList[i]}: {valueList[i]}");
        // }
        
        
    }

    #endregion
    
    
    
    
}
