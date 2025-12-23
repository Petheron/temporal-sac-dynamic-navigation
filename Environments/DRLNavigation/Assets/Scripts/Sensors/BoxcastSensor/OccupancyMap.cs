using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using UnityEditor;
using UnityEngine;



public class OccupancyMap:MonoBehaviour
{
    [Header("Grid Settings")] 
    public OccupancyGridProperties occupancyGridProperties;

    private OccupancyGridDataType occupancyGridDataType = OccupancyGridDataType.VolumetricGrid;
    private int gridSize;
    private float cellSize;
    private int levelsOfDetail=3;
    private float cellSizePercentageBuffer;
    private LayerMask colliderMask;
    
    private VolumetricBaseGrid baseGrid;
    private int baseNumCells;
    private Transform rootTransform;
    [SerializeField,ReadOnly] private VisionTag visionTags;
    [SerializeField,ReadOnly] private MutatorType visibleMutatorTypes;
    private VisibleTags visibleTags;
    private bool useTagIndicesForVoxelValue;
    [SerializeField,ReadOnly] private int numDetectableTags;
    [SerializeField,ReadOnly] private int[] detectableValues;
    private EnvironmentParameters environmentParameters;

    [Header("Testing Variables")] 
    public int testLODIndex =0;
    public int testNumChannels =3;
    public Vector3  testSamplePosition;
    public int testVoxelIndex = 0;
    public int testVoxelCount0 = 0;
    public float testSampleRadius = 3.0f;
    public Vector3Int testSampleSize = 3*Vector3Int.one;
    public Vector3 testSampleEulerRotation;
    [SerializeField] private int[] TestSampleMap;
    [SerializeField] private int[] TestSampleIndexToCellIndex;
    public Transform testTransform;
    public int testTransformIndex;
    public bool showGizmos = true;
    public bool permanentlyShowGizmos = true;
    public bool showTestGizmos = true;

    public MasterController.ControllerEvent OccupancyMapInitialized;
    public MasterController.ControllerEvent ReadyEvent;
    public bool IsReady { get; private set; }
    [HideInInspector] public bool isMapInitialized = false;

    private void InitializeGrid()
    {
        gridSize = Mathf.NextPowerOfTwo(gridSize);

        rootTransform = transform;
        numDetectableTags = visibleTags.GetVisibleTagsCount();
        detectableValues = useTagIndicesForVoxelValue? visibleTags.GetAllVisibleTagIndices() : visibleTags.GetAllVisibleValues();

        baseGrid = new VolumetricBaseGrid(gridSize, cellSize, levelsOfDetail, numDetectableTags, rootTransform.position,detectableValues);
        baseNumCells = baseGrid.GetBaseNumCells();
    }
    
    private void GetPropertiesFromScriptableObject()
    {
        if (!occupancyGridProperties)
        {
            Debug.LogError("No Occupancy Grid Properties Assigned");
            return;
        }

        occupancyGridDataType = occupancyGridProperties.occupancyGridDataType;
        gridSize = occupancyGridProperties.gridSize;
        cellSize = occupancyGridProperties.cellSize;
        levelsOfDetail = occupancyGridProperties.levelsOfDetail;
        cellSizePercentageBuffer = occupancyGridProperties.cellSizePercentageBuffer;
        colliderMask = occupancyGridProperties.colliderMask;

        useTagIndicesForVoxelValue = occupancyGridProperties.useTagIndicesForVoxelValue;
        visibleTags = occupancyGridProperties.VisibleTags;
    }

    private void GetPropertiesFromEnvParameters()
    {
        if (environmentParameters == null) return;
        
        int visionTagMask = (int) environmentParameters.GetWithDefault("visible_layer_mask",-1);
        int mutatorTypeMask = (int) environmentParameters.GetWithDefault("visible_mutator_mask",-1);
        
        visibleTags.SetDataFromBitMasks(visionTagMask, mutatorTypeMask);
        visionTags = visibleTags.GetVisionTags();
        visibleMutatorTypes = visibleTags.GetVisibleMutatorTypes();
        
        gridSize = (int) environmentParameters.GetWithDefault("occupancy_map_grid_size", gridSize);
        levelsOfDetail = (int) environmentParameters.GetWithDefault("occupancy_map_levels_of_detail", levelsOfDetail);
        cellSize = environmentParameters.GetWithDefault("occupancy_map_cell_size", cellSize);
        cellSizePercentageBuffer = environmentParameters.GetWithDefault("occupancy_map_cell_size_buffer", cellSizePercentageBuffer);
    }


    public void InitializeOccupancyMap()
    {
        isMapInitialized = false;
        GetPropertiesFromScriptableObject();
        GetPropertiesFromEnvParameters();
        InitializeGrid();
        Perceive();
        
        OccupancyMapInitialized?.Invoke();
    }

    public void Initialize()
    {
        environmentParameters = Academy.Instance.EnvironmentParameters;
        isMapInitialized = false;
        IsReady = true;
        ReadyEvent?.Invoke();
    }

    public Quaternion GetGridOrientation()
    {
        return rootTransform ? rootTransform.rotation : Quaternion.identity;
    }
    
    /// <summary>
    /// Gets the global position of the cell centre for use in a boxcast.
    /// </summary>
    /// <param name="cellIndex">The index of the considered cell. </param>
    /// <returns></returns>
    public Vector3 GetCellGlobalPosition(int cellIndex, int LODIndex=0)
    {
        if (baseGrid == null) return Vector3.zero;
        return baseGrid.GetCellGlobalPosition(cellIndex, LODIndex);
    }

    public Vector3Int GetDimIndexFromGlobalPosition(Vector3 position, int LODIndex=0)
    {
        if (baseGrid == null) return Vector3Int.zero;
        return baseGrid.GetDimsFromGlobalPosition(position, LODIndex);
    }

    public int[] GetAllDimIndicesFromGlobalPosition(Vector3 position)
    {
        List<int> indices = new List<int>();

        for (int l = 0; l < levelsOfDetail; l++)
        {
            Vector3Int dimIdx= GetDimIndexFromGlobalPosition(position, l);
            for (int i = 0; i < 3; i++)
            {
                indices.Add(dimIdx[i]);
            }
        }

        return indices.ToArray();
    }

    private int GetIndexFromGlobalPosition(Vector3 position, int LODIndex = 0)
    {
        return baseGrid.GetGridIndexFromGlobalPosition(position, LODIndex);
    }

    public int[] GetAllIndicesFromGlobalPosition(Vector3 position)
    {
        List<int> indices = new List<int>();
        for (int i = 0; i < levelsOfDetail; i++)
        {
            indices.Add(GetIndexFromGlobalPosition(position,i));
        }
        return indices.ToArray();
    }

    public int GetValueFromGlobalPosition(Vector3 position, int lodIndex = 0)
    {
        int idx = GetIndexFromGlobalPosition(position, lodIndex);
        return baseGrid.GetVoxel(idx, lodIndex).Value();
    }

    public int GetValue(int i, int j, int k, int lodIndex = 0)
    {
        return baseGrid.GetVoxel(baseGrid.GetGridIndex(i, j, k, lodIndex), lodIndex).Value();
    }

    public int GetValue(int idx, int lodIndex = 0)
    {
        return baseGrid.GetVoxel(idx, lodIndex).Value();
    }
    
    public int GetTotalCountFromGlobalPosition(Vector3 position, int lodIndex = 0)
    {
        int idx = GetIndexFromGlobalPosition(position, lodIndex);
        return baseGrid.GetVoxel(idx, lodIndex).Data.TotalCount();
    }

    public int GetGridIndex(int i, int j, int k, int lod = 0)
    {
        return baseGrid.GetGridIndex(i, j, k, lod);
    }
    
    /// <summary>
    /// Perceive the environment and store into an array.
    /// Calculates cell global position, casts box cast at this position and stores the objects overlapped. 
    /// </summary>
    private void Perceive()
    {
        Voxel[] baseVoxelArray = new Voxel[gridSize * gridSize * gridSize];
        Vector3 halfExtents = 0.5f*cellSize * (1f - 0.01f*cellSizePercentageBuffer)*Vector3.one;
        for (int i = 0; i < baseNumCells; i++)
        {
            Vector3 cellCentre = baseGrid.GetGlobalPositionFromIndex(i);
            Quaternion orientation = Quaternion.identity;
            Collider[] colliders = Physics.OverlapBox(cellCentre, halfExtents, orientation, colliderMask,
                QueryTriggerInteraction.Collide);
            
            // Collect tags of all collided objects if not collected already
            int[] tagsCount = new int[numDetectableTags];
            foreach (var col in colliders)
            {
                int idx = visibleTags.IndexFromString(col.tag);
                if(idx<0) continue;
                if(idx>numDetectableTags) continue;
                tagsCount[idx] += 1;
            }
            
            Voxel v = new Voxel(baseGrid.GetLocalPosition(i), cellSize, new VoxelData(numDetectableTags, detectableValues));
            v.Data.SetCounts(tagsCount);
            baseVoxelArray[i] = v;
        }
        
        baseGrid.Initialize(ref baseVoxelArray);

        isMapInitialized = true;
    }

    public void UpdateDynamicOccupancyIndex(int[] oldIdx, int[] newIdx, VisionTag dynamicTag, MutatorType mutatorType)
    {
        baseGrid?.UpdateDynamicCount(oldIdx, newIdx, ComputeTagValue(dynamicTag, mutatorType));
    }

    public void AddNewDynamicOccupancyIndex(int[] idxs, VisionTag dynamicTag, MutatorType mutatorType)
    {
        baseGrid?.AddNewDynamicCount(idxs,ComputeTagValue(dynamicTag, mutatorType));
    }
    
    public void SubtractDynamicOccupancyIndex(int[] idxs, VisionTag dynamicTag, MutatorType mutatorType)
    {
        baseGrid?.SubtractDespawningDynamicCount(idxs,ComputeTagValue(dynamicTag, mutatorType));
    }

    private int ComputeTagValue(VisionTag dynamicTag, MutatorType mutatorType)
    {
        if (mutatorType != MutatorType.None)
        {
            return visibleTags.GetValue(mutatorType, useTagIndicesForVoxelValue);
        }
        return visibleTags.GetValue(dynamicTag, useTagIndicesForVoxelValue);
    }

    
    public void SampleOccupancyMap(Vector3 position, Vector3Int sampleSize, out int[] samples, out int[] indices, int initialLODIndex=0, int numChannels=1)
    {
        baseGrid.Sample(position, sampleSize, out int[] gridSamples, out int[] gridIndices, initialLODIndex, numChannels);
        samples = gridSamples;
        indices = gridIndices;
    }

    public float CellSize(int LODIndex = 0)
    {
        if (baseGrid == null) return cellSize;
        return baseGrid.CellSize(LODIndex);
    }


    public void TestSampling()
    {
        SampleOccupancyMap(testSamplePosition, testSampleSize,out TestSampleMap, out TestSampleIndexToCellIndex, 0,testNumChannels);
    }

    public void TestIndexFromLocation()
    {
        testTransformIndex = baseGrid.GetGridIndexFromGlobalPosition(testTransform.position, testLODIndex);
    }
    
    

#if UNITY_EDITOR
    
    private void SampleSphere()
    {
        Vector3 centre = testSamplePosition;
        float radius = testSampleRadius;
        int r = (int) radius;

        if (baseGrid == null) return;
        Vector3Int centreDims = baseGrid.GetDimsFromGlobalPosition(centre, testLODIndex);
        Vector3 centreCellPos = GetCellGlobalPosition(baseGrid.GetGridIndex(centreDims.x,centreDims.y,centreDims.z,testLODIndex), testLODIndex);

        Vector3Int size = new Vector3Int(testSampleSize.x/2,testSampleSize.y/2,testSampleSize.z/2);
        
        for (int i = -size.x; i <= size.x; i++)
        {
            for (int j = -size.y; j <= size.y; j++)
            {
                for (int k = -size.z; k <= size.z; k++)
                {
                    List<float> ijk = new List<float> {Mathf.Abs(i), Mathf.Abs(j), Mathf.Abs(k)};
                    float cubeHalfSize = ijk.Max();
                    // Calculate global position for sample (i,j,k) relative to agent position & rotation.
                    Vector3 indexOffset = new Vector3(i, j, k);
                    Vector3 baseOffset = indexOffset*cellSize;
                    Vector3 preRotateOffset = baseOffset;

                    Vector3 eulerAngles = testSampleEulerRotation;
                    
                    Quaternion rot = Quaternion.Euler(eulerAngles);
                    
                    Vector3 offset = Matrix4x4.Rotate(rot).MultiplyVector(preRotateOffset);

                    Vector3 pos = centreCellPos + offset;
                    
                    // Access OGM cell at this position and obtain cell position
                    Vector3Int cellDimIndex = baseGrid.GetDimsFromGlobalPosition(pos, testLODIndex);
                    int cellIndex = baseGrid.GetGridIndex(cellDimIndex.x,cellDimIndex.y,cellDimIndex.z);
                    Vector3 cellPos = GetCellGlobalPosition(cellIndex);

                    // Determine debugging colors and draw gizmos/handles
                    float floatR = r;
                    Color color = new Color(floatR+i, floatR+j, floatR+k);
                    color /= 2 * floatR;
                    color.a = 1f;
                    Gizmos.color = color;
                    Gizmos.DrawWireCube(cellPos,0.9f*Vector3.one);
                    Gizmos.DrawSphere(pos,0.1f);
                    DrawString($"({i}, {j}, {k})", cellPos, Color.white, Vector2.zero);
                    DrawString($"{offset.ToString()}", pos, color, 2*Vector2.down);

                }
            }
        }
    }
    
    private static void DrawString(string text, Vector3 worldPosition, Color textColor, Vector2 anchor, float textSize = 15f)
    {
        var view = SceneView.currentDrawingSceneView;
        if (!view)
            return;
        Vector3 screenPosition = view.camera.WorldToScreenPoint(worldPosition);
        if (screenPosition.y < 0 || screenPosition.y > view.camera.pixelHeight || screenPosition.x < 0 || screenPosition.x > view.camera.pixelWidth || screenPosition.z < 0)
            return;
        var pixelRatio = HandleUtility.GUIPointToScreenPixelCoordinate(Vector2.right).x - HandleUtility.GUIPointToScreenPixelCoordinate(Vector2.zero).x;
        Handles.BeginGUI();
        var style = new GUIStyle(GUI.skin.label)
        {
            fontSize = (int)textSize,
            normal = new GUIStyleState() { textColor = textColor }
        };
        Vector2 size = style.CalcSize(new GUIContent(text)) * pixelRatio;
        var alignedPosition =
            ((Vector2)screenPosition +
             size * ((anchor + Vector2.left + Vector2.up) / 2f)) * (Vector2.right + Vector2.down) +
            Vector2.up * view.camera.pixelHeight;
        GUI.Label(new Rect(alignedPosition / pixelRatio, size / pixelRatio), text, style);
        Handles.EndGUI();
    }

    private void DrawOccupancyMap(Vector3 cubeSize)
    {
        if (baseGrid == null) return;
        
        float edgeAlpha = occupancyGridProperties.edgeAlpha;

        for (int i = 0; i < baseGrid.NumCells(testLODIndex) ; i++)
        {
            int value = baseGrid.GetVoxel(i, testLODIndex).Value();
            if (value <= 0) continue;

            Color volumeColor = occupancyGridProperties.GetColor(value);
            Color edgeColor = new Color(volumeColor.r,volumeColor.g,volumeColor.b,edgeAlpha);
            Vector3 cellCentre = GetCellGlobalPosition(i, testLODIndex);
            
            Gizmos.matrix = Matrix4x4.TRS(Vector3.zero,GetGridOrientation(),Vector3.one);
            Gizmos.color = volumeColor;
            Gizmos.DrawCube(cellCentre, cubeSize);
            Gizmos.color = edgeColor;
            Gizmos.DrawWireCube(cellCentre, cubeSize);
            Gizmos.matrix = Matrix4x4.identity;
        }
    }
    
    private void OnDrawGizmos()
    {
        if (!showGizmos) return;
        
        if (showTestGizmos)
        {
            SampleSphere();
        }

        if (permanentlyShowGizmos)
        {
            if (baseGrid == null) return;
            Vector3 cubeSize = baseGrid.CellSize(testLODIndex) * (1f - 0.01f * cellSizePercentageBuffer) * Vector3.one;
            DrawOccupancyMap(cubeSize);
        }
    }

    private void OnDrawGizmosSelected()
    {
        if (!showGizmos) return;

        if (baseGrid == null) return;

        Vector3 cubeSize = baseGrid.CellSize(testLODIndex) * (1f - 0.01f * cellSizePercentageBuffer) * Vector3.one;

        if (!permanentlyShowGizmos)
        {
            DrawOccupancyMap(cubeSize);
        }


        if (TestSampleIndexToCellIndex.Length > 0)
        {
            int numSample = testSampleSize.x * testSampleSize.y * testSampleSize.z;

            for (int i = 0; i < numSample; i++)
            {
                int index = TestSampleIndexToCellIndex[testLODIndex + i*testNumChannels];
                Vector3 cellCentre = GetCellGlobalPosition(index, testLODIndex);
                Quaternion orientation = GetGridOrientation();
                float b = 1f/numSample * i;
                Color highlightColor = new Color(1f, 0.9f, b, 0.1f);
                Color borderColor = new Color(highlightColor.r,highlightColor.g,highlightColor.b, 0.7f);
                Gizmos.color = highlightColor;
                Gizmos.matrix = Matrix4x4.TRS(Vector3.zero,orientation,Vector3.one);
                Gizmos.DrawCube(cellCentre, cubeSize);
                Gizmos.color = borderColor;
                Gizmos.DrawWireCube(cellCentre, cubeSize);
                Gizmos.matrix = Matrix4x4.identity;
            }
        }


        if (testVoxelIndex >= 0)
        {
            testVoxelCount0 = baseGrid.GetVoxel(testVoxelIndex, 0).Counts()[0];
            Vector3 pos = GetCellGlobalPosition(testVoxelIndex, testLODIndex);
            Quaternion or = GetGridOrientation();
            Color hc = new Color(1f, 0.9f, 0.5f, 0.1f);
            Color bc = new Color(hc.r,hc.g,hc.b, 0.7f);
            Gizmos.color = hc;
            Gizmos.matrix = Matrix4x4.TRS(Vector3.zero,or,Vector3.one);
            Gizmos.DrawCube(pos, cubeSize);
            Gizmos.color = bc;
            Gizmos.DrawWireCube(pos, cubeSize);
            Gizmos.matrix = Matrix4x4.identity; 
        }
    }
#endif
    
}
