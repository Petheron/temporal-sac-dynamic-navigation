using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

namespace LevelGeneration
{
    public enum GeneratorType
    {
        PERLIN,
        WFC
    }
    
    public class Chunk:MonoBehaviour
    {
        public GeneratorType generatorType = GeneratorType.PERLIN;
        public float cellSize = 1.0f;
        public Vector3Int gridSize;
        private Vector3Int newGridSize;
        public AdditionalNoiseParameters noiseParameters;
        private NoiseGenerator generator;

        private List<Vector3> generatedVertices;
        private List<Vector3> generatedNormals;
        private List<int> generatedTriangles;
        private List<Vector2> generatedUV;
        
        public byte[,,] grid;
        public MeshFilter meshFilter;
        public MeshCollider meshCollider;
        [HideInInspector] public OccupancyMap occupancyMap;
        [SerializeField] private List<int[]> indices;

        private Vector3[] vertexPositionOffsets = new Vector3[]
        {
            new (-1, 1, -1),
            new (-1, 1, 1),
            new (1, 1, 1),
            new (1, 1, -1),
            new (-1, -1, -1),
            new (-1, -1, 1),
            new (1, -1, 1),
            new (1, -1, -1),
        };

        // 4 triangles indices, 3 face check offset, 2 uv offset
        private int[][] faces = new int[6][]
        {
            new []{0, 1, 2, 3, 0, 1, 0, 0, 0},    //top
            new []{7, 6, 5, 4, 0,-1, 0, 1, 0},   //bottom
            new []{3, 2, 6, 7, 1, 0, 0, 1, 1},    //right
            new []{1, 0, 4, 5,-1, 0, 0, 1, 1},   //left
            new []{2, 1, 5, 6, 0, 0, 1, 1, 1},    //back
            new []{0, 3, 7, 4, 0, 0,-1, 1, 1}    //front
        };

        private Vector3[] faceNormals = 
        {
            new (0,1,0),
            new (0,-1,0),
            new (1,0,0),
            new (-1,0,0),
            new (0,0,1),
            new (0,0,-1),
        };

        private void SetOccupancyMap(OccupancyMap map)
        {
            occupancyMap = map;
            if (occupancyMap)
            {
                occupancyMap.OccupancyMapInitialized += SetTag;
                if (occupancyMap.isMapInitialized)
                {
                    SetTag();
                }
            }
        }

        public void Initialize(GeneratorType generatorType, float cellSize, Vector3Int gridSize, AdditionalNoiseParameters noiseParameters, OccupancyMap occupancyMap)
        {
            this.generatorType = generatorType;
            this.cellSize = cellSize;
            this.gridSize = gridSize;
            this.noiseParameters = noiseParameters;
            SetOccupancyMap(occupancyMap);
        }

        public void Generate(Vector3Int gridSize, AdditionalNoiseParameters noiseParameters)
        {
            newGridSize = gridSize;
            this.noiseParameters = noiseParameters;
            Generate();
        }

        public void Generate()
        {
            // Subtract old values from occupancy map
            if (grid != null)
            {
                GetIndices();
                SubtractValuesFromOccupancyMap();
            }
            
            // Resize grid if required
            if (newGridSize != Vector3Int.zero)
            {
                gridSize = newGridSize;
                newGridSize = Vector3Int.zero;
            }
            
            // create new grid
            generator = generatorType == GeneratorType.PERLIN ? new NoiseGenerator() : new WaveFunctionCollapse();
            grid = generator.Generate(gridSize.x, gridSize.y, gridSize.z, transform.position, cellSize, noiseParameters);
            
            // insert new grid values into occupancy map
            GetIndices();
            AddValuesToOccupancyMap();
            
            // generate voxel visuals
            GenerateBoxels();
        }

        /// <summary>
        /// Generate chunk according to input grid.
        /// </summary>
        /// <param name="inputGrid"></param>
        public void Generate(byte[] inputGrid)
        {
            grid = new byte[gridSize.x, gridSize.y, gridSize.z];
            for (int i = 0; i < gridSize.x; i++)
            {
                for (int j = 0; j < gridSize.y; j++)
                {
                    for (int k = 0; k < gridSize.z; k++)
                    {
                        grid[i, j, k] = inputGrid[i + k * gridSize.x + j * gridSize.x * gridSize.z];
                    }
                }
            }
            
            GenerateBoxels();
        }

        /// <summary>
        /// Construct the mesh and collision geometry for the chunk.
        /// </summary>
        private void GenerateBoxels()
        {
            if (grid == null) return;
            if (grid.Length != gridSize.x * gridSize.y * gridSize.z) return;
            if (!meshFilter) meshFilter = GetComponent<MeshFilter>();
            if (!meshCollider)
            {
                meshCollider = GetComponent<MeshCollider>();
            }

            generatedVertices = new List<Vector3>();
            generatedNormals = new List<Vector3>();
            generatedTriangles = new List<int>();
            generatedUV = new List<Vector2>();

            for (int i = 0; i < gridSize.x; i++)
            {
                for (int j = 0; j < gridSize.y; j++)
                {
                    for (int k = 0; k < gridSize.z; k++)
                    {
                        if (IsCellEmpty(i,j,k)) continue;

                        Vector3 pos = new Vector3(cellSize * i, cellSize * j, cellSize * k);

                        for (int l = 0; l < faces.Length; l++)
                        {
                            var face = faces[l];
                            if (IsCellEmpty(i+face[4], j+face[5], k+face[6]))
                            {
                                AddQuad(pos,face[0],face[1],face[2],face[3], generatedVertices.Count, l);
                            }
                        }
                    }
                }
            }

            if (!meshFilter) return;

            meshFilter.mesh = new Mesh()
            {
                vertices = generatedVertices.ToArray(),
                triangles = generatedTriangles.ToArray(),
                uv = generatedUV.ToArray(),
                normals = generatedNormals.ToArray(),
            };

            meshCollider.sharedMesh = new Mesh()
            {
                vertices = generatedVertices.ToArray(),
                triangles = generatedTriangles.ToArray()
            };
        }

        /// <summary>
        /// Get the occupancy map indices that the chunk inhabits.
        /// </summary>
        private void GetIndices()
        {
            if (!occupancyMap) return;
            if (!occupancyMap.isMapInitialized) return;
            
            indices = new List<int[]>();
            Vector3 cornerPos = transform.position;

            for (int i = 0; i < gridSize.x; i++)
            {
                for (int j = 0; j < gridSize.y; j++)
                {
                    for (int k = 0; k < gridSize.z; k++)
                    {
                        if(grid[i,j,k]==0) continue;
                        Vector3 pos = new Vector3(cellSize * i, cellSize * j, cellSize * k) + cornerPos;
                        int[] idx = occupancyMap.GetAllIndicesFromGlobalPosition(pos);
                        indices.Add(idx);
                    }
                }
            }
        }

        private void SubtractValuesFromOccupancyMap()
        {
            if (!occupancyMap) return;
            if (indices == null) return;
            if (!occupancyMap.isMapInitialized) return;
            foreach (var idx in indices)
            {
                occupancyMap.SubtractDynamicOccupancyIndex(idx,VisionTag.Solid);
            }

            
        }
        
        private void AddValuesToOccupancyMap()
        {
            if (!occupancyMap) return;
            if(indices==null) return;
            if (!occupancyMap.isMapInitialized) return;
            
            foreach (var idx in indices)
            {
                occupancyMap.AddNewDynamicOccupancyIndex(idx,VisionTag.Solid);
            }
        }

        private bool IsCellNull(int x, int y, int z)
        {
            if (!IsCellInGrid(x, y, z)) return true;
            return grid[x, y, z] == Byte.MaxValue;
        }

        private bool IsCellEmpty(int x, int y, int z)
        {
            if (!IsCellInGrid(x, y, z)) return true;
            if (IsCellNull(x, y, z)) return true;
            return grid[x, y, z] == 0;
        }

        private bool IsCellInGrid(int x, int y, int z)
        {
            if (x < 0 || x >= gridSize.x) return false;
            if (y < 0 || y >= gridSize.y) return false;
            if (z < 0 || z >= gridSize.z) return false;
            return true;
        }

        private void AddQuad(Vector3 pos, int aIdx, int bIdx, int cIdx, int dIdx, int i, int faceNum)
        {
            var a = vertexPositionOffsets[aIdx];
            var b = vertexPositionOffsets[bIdx];
            var c = vertexPositionOffsets[cIdx];
            var d = vertexPositionOffsets[dIdx];
            generatedVertices.AddRange(new List<Vector3>(){pos+a*cellSize/2,pos+b*cellSize/2,pos+c*cellSize/2,pos+d*cellSize/2});
            generatedTriangles.AddRange(new List<int>(){i,i+1,i+2,i,i+2,i+3});
            
            var faceNormal = faceNormals[faceNum];
            for (int j = 0; j < 4; j++)
            {
                generatedNormals.Add(faceNormal);
            }
            
            Vector2 bottomLeft = new Vector2(faces[faceNum][7], faces[faceNum][8]) / 2f;
            
            generatedUV.AddRange(new List<Vector2>{bottomLeft+0.5f*Vector2.up, bottomLeft+0.5f*Vector2.one, bottomLeft+0.5f*Vector2.right, bottomLeft});
        }

        private void Awake()
        {
            gameObject.tag = "Untagged";
        }

        private void SetTag()
        {
            gameObject.tag = "Solid";
            Generate();
        }

        private void OnDisable()
        {
            if (!occupancyMap) return;
            occupancyMap.OccupancyMapInitialized -= SetTag;
        }
    }
}