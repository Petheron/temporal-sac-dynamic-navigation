using UnityEngine;

public class BaseGrid
{
    public static Vector3Int GetOddGrid(Vector3Int gridSize)
    {
        Vector3Int output = new Vector3Int();
        for (int i = 0; i < 3; i++)
        {
            output[i] = EnsureOddDim(gridSize[i]);
        }
        return output;
    }
    
    public static int EnsureOddDim(int dim)
    {
        if (dim <= 0) return 0;
        return dim + (dim + 1) % 2;
    }
}
