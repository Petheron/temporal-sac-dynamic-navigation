using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Serialization;
using Random = UnityEngine.Random;

namespace LevelGeneration
{
    public class LevelGenerator : MonoBehaviour
    {
        [Header("Chunk Parameters")] 
        [SerializeField] private bool isCentredAtWorldOrigin;
        [SerializeField, ReadOnly] private Vector3Int chunkSize;
        [SerializeField] private Vector3Int numChunks;
        [SerializeField] private GeneratorType generatorType = GeneratorType.PERLIN;
        public float cellSize = 1.0f;

        [Header("Simplex Noise Parameters")] 
        [SerializeField] private float noiseAmplitude = 1f;
        [SerializeField] private float noiseFrequency = 1.0f;
        [SerializeField] private float seedFrequency = 1.0f;
        [Range(1, 10)] [SerializeField] private int noiseIterations = 1;
        [SerializeField] private float noiseAmplitudePersistence = 0.5f;
        private AdditionalNoiseParameters noiseParams;


        [Header("Regeneration Parameters")] 
        private int initialSeedIndex = 0;
        private int currentSeed;
        private int numChunkSeeds = 256;
        private int chunkRegenerationStepDelay = 1000;
        private int regenerationStep;
        private bool useConstantHeight;
        private int maxGenerationHeight;
        private int generationSize;
        
        [Header("Evaluation Parameters")]
        private int evaluationSeed;
        private int evaluationSeedIdx;

        private List<Chunk> chunks;
        [SerializeField, ReadOnly] private int seedIndex;
        [SerializeField, ReadOnly] private int seedCount;

        [Header("Reference Parameters")] 
        public Chunk chunkPrefab;
        [SerializeField] private NavMeshConstructor navMeshConstructor;
        private OccupancyMap occupancyMap;
        private AgentSpawner agentSpawner;
        private EnvironmentParameters environmentParams;
        [SerializeField] private LevelGeneratorProperties levelGeneratorProperties;
        [SerializeField] private AgentEvaluationManager agentEvaluationManager;

        private bool IsPerformRegenerate => chunkRegenerationStepDelay > 0
                                            && Academy.Instance.StepCount > 0
                                            && Academy.Instance.StepCount != regenerationStep
                                            && Academy.Instance.StepCount % chunkRegenerationStepDelay == 0;

        private Vector3Int GridSize => Vector3Int.Scale(chunkSize, numChunks);
        [ReadOnly, SerializeField] private Vector3Int gridSize;

        private static int[] _seeds =
        { -72, -101, -144, -179, -180, -98, -86, -102, -150, -166, 133,
            -251, 161, -173, 126, 56, -38, 200, -229, -172, 54, -190,
            -78, -120, 79, 37, -130, 59, -57, 61, 190, 187, -56,
            176, 154, -9, -70, -50, -241, 14, -206, 172, -109, -91,
            -189, -205, 127, 197, 83, 234, 141, -99, -187, -48, 220,
            -65, -1, -148, 142, 99, 113, -34, 76, 224, -176, -25,
            -51, -204, 50, 210, 132, -217, 92, -129, -12, -207, 47,
            -118, 171, -184, 166, 179, -247, 28, 70, -242, 163, 109,
            241, 231, -24, -183, 144, 65, -123, -60, -3, -195, -192,
            228, -27, -28, 131, 151, -218, -219, 182, -20, 4, -49,
            86, -159, -88, -112, 36, 225, -161, -243, -200, 191, -33,
            -162, 169, 120, -223, -67, -96, -26, -58, 6, 88, 135,
            -44, 103, -21, -31, 43, 87, -215, -135, -256, -43, 242,
            137, -132, 223, 243, 51, 107, -121, -142, 73, 45, 202,
            124, 129, -252, 41, -196, 215, 38, 205, 249, 173, 72,
            145, -17, 91, 19, 22, 130, -145, 158, -222, 116, -154,
            -19, 204, 254, 30, -97, -47, 52, 123, 55, 235, -35,
            177, 139, -170, 13, 203, 48, 25, -77, -36, 100, -90,
            -227, 252, -255, -54, 245, -185, 212, 102, -168, -73, 108,
            -224, 180, 244, -103, 53, -75, -220, -232, -152, 40, 21,
            159, -107, -22, -126, -155, -37, -74, 81, -202, 238, -140,
            236, 23, 143, -122, 248, 185, 10, -174, -23, -79, 160,
            -235, -143, 233, 94, -238, -201, -89, -134, -133, -113, -85,
            -18, -5, 74
        };

        private static int[] evaluationSeeds =
        {
            -89, -134, -133, -113, -85, -18, -5, 74
        };

        public void Initialize(EnvironmentParameters envParams, OccupancyMap occMap)
        {
            environmentParams = envParams;
            occupancyMap = occMap;
            agentSpawner = occupancyMap.GetComponent<AgentSpawner>();
            GetEnvironmentParameters();
            if (agentEvaluationManager)
            { 
                agentEvaluationManager.OnResultsSaved += RegenerateChunks;
            }

            if (!navMeshConstructor)
            {
                navMeshConstructor = GetComponentInParent<NavMeshConstructor>();
            }
            Generate();
        }

        private void GetEnvironmentParameters()
        {
            if (environmentParams == null) return;
            chunkRegenerationStepDelay =
                (int)environmentParams.GetWithDefault("level_generation_delay", levelGeneratorProperties.LEVEL_GENERATION_DELAY);
            numChunkSeeds =
                (int)environmentParams.GetWithDefault("level_generation_num_seeds", levelGeneratorProperties.CHUNK_NUM_SEEDS);
            initialSeedIndex =
                (int)environmentParams.GetWithDefault("level_generation_initial_seed_index", levelGeneratorProperties.CHUNK_INITIAL_SEED_INDEX);

            noiseAmplitude = levelGeneratorProperties.NOISE_AMPLITUDE;
            noiseFrequency = levelGeneratorProperties.NOISE_FREQUENCY;
            seedFrequency = levelGeneratorProperties.SEED_FREQUENCY;
            noiseIterations = levelGeneratorProperties.NOISE_ITERATIONS;
            noiseAmplitudePersistence = levelGeneratorProperties.NOISE_AMPLITUDE_PERSISTENCE;
            useConstantHeight = levelGeneratorProperties.USE_CONSTANT_LEVEL_GENERATION_HEIGHT;
            generationSize = levelGeneratorProperties.LEVEL_GENERATION_SIZE;
            maxGenerationHeight = levelGeneratorProperties.LEVEL_GENERATION_HEIGHT;

            numChunks = new Vector3Int(Math.Max(numChunks.x, 1), 1, Math.Max(numChunks.z, 1));
            generationSize = (int)environmentParams.GetWithDefault("level_generation_size", generationSize);
            generationSize += (generationSize+1) % 2; // ensure generationWidth is odd
            maxGenerationHeight = Math.Max((int)environmentParams.GetWithDefault("level_generation_max_height", maxGenerationHeight),1);
            useConstantHeight = environmentParams.GetWithDefault("level_generation_use_constant_height", useConstantHeight?1:0)>0;
            
            SetChunkSize();
            
            // Ensure number of maps is within number of seeds
            if (numChunkSeeds > _seeds.Length) numChunkSeeds = _seeds.Length;
            // Wrap negative initial seed index around array
            if (initialSeedIndex < 0) initialSeedIndex = _seeds.Length - Math.Abs(initialSeedIndex);
        }

        private bool IsEvaluating()
        {
            if (Academy.Instance.IsCommunicatorOn) return false;
            if (!agentSpawner) return false;
            return agentSpawner.IsEvaluating;
        }

        private void SetCurrentSeed()
        {
            // print($"Eval Index: {evaluationSeedIdx}, Training Idx: {seedIndex}");
            currentSeed = IsEvaluating() ? evaluationSeeds[evaluationSeedIdx] : _seeds[seedIndex];
            // print($"Current Seed: {currentSeed}");
        }

        private void SetChunkSize()
        {
            int width = Mathf.CeilToInt((float)generationSize / numChunks.x);
            int depth = Mathf.CeilToInt((float)generationSize / numChunks.z);
            int height = useConstantHeight ? maxGenerationHeight : (Mathf.Abs(currentSeed)*137 + 11)  % maxGenerationHeight + 1;
            height = Mathf.CeilToInt((float)height / numChunks.y);
            chunkSize = new Vector3Int(width, height, depth);
        }

        private void FixedUpdate()
        {
            if(IsPerformRegenerate) RegenerateChunks();
        }

        public void Generate()
        {
            gridSize = GridSize;
            seedIndex = initialSeedIndex;
            seedCount = 0;
            SetCurrentSeed();
            SetChunkSize();
            RefreshNoiseParams(currentSeed);
            ResetChunks();
            CreateChunks();
            ConstructNavMesh();
        }

        public void Generate(byte[] inputGrid, Vector3Int inputGridSize)
        {
            for (int i = 0; i < 3; i++)
            {
                chunkSize[i] = inputGridSize[i] / Math.Max(numChunks[i],1);
            }
            
            gridSize = GridSize;
            RefreshNoiseParams(currentSeed);

            ResetChunks();
            CreateChunksFromInputGrid(inputGrid);
        }
        

        public void ClearChunks()
        {
            ResetChunks();
            if (transform.childCount == 0) return;
            while (transform.childCount > 0)
            {
                DestroyImmediate(transform.GetChild(0).gameObject);
            }
        }

        private void ResetChunks()
        {
            chunks ??= new List<Chunk>();

            if (chunks.Count > 0)
            {
                foreach (var chunk in chunks)
                {
                    Destroy(chunk.gameObject);
                }
                chunks.Clear();
            }
        }

        private void CreateChunks()
        {
            var cornerPosition = isCentredAtWorldOrigin
                ? new Vector3(-cellSize*gridSize.x/2f, -cellSize*gridSize.y/2f, -cellSize*gridSize.z/2f)
                : Vector3.zero;

            // Ensure (0,0,0) is centred in a cell
            Vector3 cornerOffset = Vector3.zero;
            for (int i = 0; i < 3; i++)
            {
                cornerOffset[i] = gridSize[i] % 2 == 0 ? 0f : cellSize / 2f;
            }
            cornerPosition += cornerOffset;
            
            // Generate Chunks
            for (int i = 0; i < numChunks.x; i++)
            {
                float x = chunkSize.x * cellSize * i;

                for (int j = 0; j < numChunks.y; j++)
                {
                    float y = chunkSize.y * cellSize * j;
                    
                    for (int k = 0; k < numChunks.z; k++)
                    {
                        float z = chunkSize.z * cellSize * k;
                        
                        Vector3 chunkPositionOffset = cornerPosition + new Vector3(x, y, z);
                        Chunk chunk = Instantiate(chunkPrefab, chunkPositionOffset, Quaternion.identity, transform);
                        chunk.gameObject.name = "Chunk (" + i + "," + j + "," + k + ")";
                        chunk.Initialize(generatorType, cellSize, chunkSize, noiseParams, occupancyMap);
                        chunk.Generate();
                        chunks.Add(chunk);
                    }
                }
            }
        }

        
        /// <summary>
        /// Generate a chunk from a given byte array of appropriate size.
        /// </summary>
        private void CreateChunksFromInputGrid(byte[] inputGrid)
        {
            var cornerPosition = isCentredAtWorldOrigin
                ? new Vector3(-cellSize*gridSize.x/2f, -cellSize*gridSize.y/2f, -cellSize*gridSize.z/2f)
                : Vector3.zero;
            
            // Ensure (0,0,0) is centred in a cell
            Vector3 cornerOffset = Vector3.zero;
            for (int i = 0; i < 3; i++)
            {
                cornerOffset[i] = gridSize[i] % 2 == 0 ? 0f : cellSize / 2f;
            }
            cornerPosition += cornerOffset;
            
            for (int i = 0; i < numChunks.x; i++)
            {
                float x = chunkSize.x * cellSize * i;

                for (int j = 0; j < numChunks.y; j++)
                {
                    float y = chunkSize.y * cellSize * j;
                    
                    for (int k = 0; k < numChunks.z; k++)
                    {
                        float z = chunkSize.z * cellSize * k;
                        
                        Vector3 chunkPositionOffset = cornerPosition + new Vector3(x, y, z);
                        Chunk chunk = Instantiate(chunkPrefab, chunkPositionOffset, Quaternion.identity, transform);
                        chunk.gameObject.name = "Chunk (" + i + "," + j + "," + k + ")";
                        chunk.Initialize(generatorType,cellSize,chunkSize,noiseParams,occupancyMap);
                        chunk.Generate(SubSampleGrid(inputGrid, i, j, k));
                        chunks.Add(chunk);
                    }
                }
            }
        }

        private byte[] SubSampleGrid(byte[] grid, int chunkIndexX, int chunkIndexY, int chunkIndexZ)
        {
            byte[] subGrid = new byte[chunkSize.x * chunkSize.y * chunkSize.z];
            int chunkOffsetX = chunkIndexX * chunkSize.x;
            int chunkOffsetZ = chunkIndexZ * chunkSize.z;
            int chunkOffsetY = chunkIndexY * chunkSize.y;
            for (int i = 0; i < chunkSize.x; i++)
            {
                for (int j = 0; j < chunkSize.y; j++)
                {
                    for (int k = 0; k < chunkSize.z; k++)
                    {
                        int subGridIndex = i + chunkSize.x * k + chunkSize.x * chunkSize.z * j;
                        int gridIndex = (chunkOffsetX + i) + 
                                        (chunkOffsetZ + k) * gridSize.x +
                                        (chunkOffsetY + j) * gridSize.x * gridSize.z;
                        subGrid[subGridIndex] = grid[gridIndex];
                    }
                }
            }
            return subGrid;
        }

        private void RegenerateChunks()
        {
            if (chunks == null) return;
            
            regenerationStep = Academy.Instance.StepCount;
            // seedIndex = (seedIndex + 1) % numChunkSeeds;
            
            seedCount = (seedCount + 1) % numChunkSeeds;
            seedIndex = (initialSeedIndex + seedCount) % _seeds.Length;
            
            
            if (IsEvaluating())
            {
                evaluationSeedIdx = (evaluationSeedIdx + 1) % evaluationSeeds.Length;
                
                if (evaluationSeedIdx == evaluationSeeds.Length - 1 && agentEvaluationManager)
                {
                    agentEvaluationManager.OnResultsSaved -= RegenerateChunks;
                }
            }
            SetCurrentSeed();
            SetChunkSize();
            RefreshNoiseParams(currentSeed);
            
            // Debug.Log($"\nRegenerating Chunks (idx: {seedIndex}, seed: {currentSeed}): \nStep Count: {Academy.Instance.StepCount} with Regen Delay: {chunkRegenerationStepDelay}\n");
            
            TimeScaleControl.TogglePauseState(true);
            
            agentSpawner?.InterruptAgents();
            
            foreach (var chunk in chunks)
            {
                chunk.Generate(chunkSize, noiseParams);
            }
            ConstructNavMesh();
        }

        private void RefreshNoiseParams(int seed)
        {
            noiseParams = new AdditionalNoiseParameters(seed, noiseAmplitude, noiseFrequency, noiseIterations,
                noiseAmplitudePersistence,seedFrequency);
        }

        private void ConstructNavMesh()
        {
            if (!navMeshConstructor) return;
            navMeshConstructor.BuildNavMesh();
        }
    }
}