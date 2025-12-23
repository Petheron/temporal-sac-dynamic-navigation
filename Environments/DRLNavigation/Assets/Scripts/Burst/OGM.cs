using System;
using System.Collections;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;
using Random = UnityEngine.Random;
using static Unity.Mathematics.math;
using float3x3 = Unity.Mathematics.float3x3;
using quaternion = Unity.Mathematics.quaternion;

public class OGM : MonoBehaviour
{
    [SerializeField] private Transform prefab;
    [SerializeField] private float oddsFilled = 0.25f;
    [SerializeField] private int gridSize = 10;
    [SerializeField] private float cellSize = 1.0f;

    private NativeArray<CellData> grid;
    [SerializeField] private Mesh mesh;
    [SerializeField] private Material material;
    
    private static readonly int Value = Shader.PropertyToID("_Value");
    private static readonly int Colors = Shader.PropertyToID("_Colors");

    private ComputeBuffer colorsBuffer;

    private Transform[] children;
    private Coroutine spawnChildrenCoroutine;
    private bool spawnChildren;

    [BurstCompile(FloatPrecision.Standard, floatMode:FloatMode.Fast, CompileSynchronously = true)]
    struct CreateVoxelJob : IJobFor
    {
        [WriteOnly]
        public NativeArray<CellData> grid;
        public float oddsFilled;
        public float gridSize;
        public float cellSize;
        public float3 parentPosition;
        public quaternion parentRotation;

        public void Execute(int i)
        {
            CellData data = new CellData()
            {
                position = parentPosition + mul(parentRotation,cellSize*float3(i % gridSize, (i / (gridSize * gridSize)) % gridSize, (i / gridSize) % gridSize)),
                rotation = parentRotation
            };
            data.value = noise.snoise(data.position) < oddsFilled ? 1 : 0;
            grid[i] = data;
        }
    }

    private struct CellData
    {
        public int value;
        public float3 position;
        public quaternion rotation;
    }

    private float3 ComputeGridLocalPosition(int i)
    {
        // idx = i + Xk + XZj
        int x = i % gridSize;
        int y = (i/(gridSize*gridSize)) % gridSize;
        int z = (i / gridSize) % gridSize;
        return new float3(x, y, z) * cellSize;
    }

    private IEnumerator SpawnChildren()
    {
        for (int i = 0; i < grid.Length; i++)
        {
            var child = Instantiate(prefab, grid[i].position, Quaternion.identity, transform);
            child.GetComponent<MeshRenderer>().material.SetFloat(Value,grid[i].value);
            children[i] = child;
        }

        spawnChildrenCoroutine = null;
        return null;
    }

    private void Update()
    {
        if (spawnChildren && spawnChildrenCoroutine == null)
        {
            spawnChildren = false;
            spawnChildrenCoroutine ??= StartCoroutine(nameof(SpawnChildren));
        }
    }

    private void OnEnable()
    {
        grid = new NativeArray<CellData>(gridSize * gridSize * gridSize, Allocator.Persistent);
        children = new Transform[grid.Length];
        
        JobHandle jobHandle = default;
        
        for (int i = 0; i < grid.Length; i+=gridSize*gridSize)
        {
            jobHandle = new CreateVoxelJob()
            {
                grid = grid,
                gridSize = gridSize,
                oddsFilled = oddsFilled,
                cellSize = cellSize,
                parentPosition = transform.position,
                parentRotation = transform.rotation
                
            }.ScheduleParallel(gridSize*gridSize*gridSize, gridSize, jobHandle);
        }
        jobHandle.Complete();
        spawnChildren = true;
        
    }

    private void OnDisable()
    {
        grid.Dispose();
        foreach (var child in children)
        {
            Destroy(child.gameObject);
        }

        children = null;
    }

    private void OnValidate()
    {
        if(!grid.IsCreated || !enabled) return;
        OnDisable();
        OnEnable();
    }
}
