using System;
using System.Collections;
using System.Collections.Generic;
using Google.Protobuf.Collections;
using Unity.VisualScripting;
using UnityEngine;
using Random = UnityEngine.Random;

namespace LevelGeneration
{
    public enum TileType
    {
        Empty,
        Solid,
        Null=255
    }
    
    public enum Direction
    {
        Forward,Right,Back,Left,Up,Down
    }

    [Serializable]
    public struct Rule
    {
        public TileType type;
        public Direction direction;
        public List<TileType> neighbors;
        public List<float> weights;
        
        public Rule(TileType t, Direction d, List<TileType> n, List<float> w)
        {
            type = t;
            direction = d;
            neighbors = n;
            weights = w;
        }
    }

    public static class RuleSet
    {
        private static Dictionary<TileType,Dictionary<Direction,List<TileType>>> validNeighbors = new Dictionary<TileType, Dictionary<Direction, List<TileType>>>()
        {
            {TileType.Empty,new Dictionary<Direction, List<TileType>>()
            {
                {Direction.Forward, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Right, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Back, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Left, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Up, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Down, new List<TileType>() {TileType.Empty,TileType.Solid}}
            }},
            
            {TileType.Solid,new Dictionary<Direction, List<TileType>>()
            {
                {Direction.Forward, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Right, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Back, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Left, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Up, new List<TileType>() {TileType.Empty,TileType.Solid}},
                {Direction.Down, new List<TileType>() {TileType.Empty,TileType.Solid}}
            }},
        };

        public static bool IsValidNeighbor(TileType value, TileType neighborValue, Direction direction)
        {
            return validNeighbors[value][direction].Contains(neighborValue);
        }

        public static int MaxNumPossibleValues()
        {
            return validNeighbors.Count;
        }
    }

    public class Tile
    {
        public bool collapsed;
        public TileType value;
        public int entropy;
        public int tileIndex;
        
        public List<TileType> possibleValues;
        public List<Tile> neighbors;
        
        public Tile(TileType[] values, int idx)
        {
            possibleValues = new List<TileType>();
            possibleValues.AddRange(values);
            entropy = possibleValues.Count;
            value = TileType.Null;
            collapsed = false;
            tileIndex = idx;
            neighbors = new List<Tile>();
        }

        public void Collapse()
        {
            collapsed = true;
            if (possibleValues.Count>1)
            {
                TileType temp = possibleValues[Random.Range(0, possibleValues.Count)];
                possibleValues.Clear();
                possibleValues.Add(temp);
            }
            value = possibleValues[0];
            entropy = 1;
            
            Debug.Log($"Tile: {tileIndex} collapsed to {value}.");
            
            UpdateNeighbors();
        }

        public void AddNeighbor(Tile neighbor)
        {
            neighbors.Add(neighbor);
        }

        public void UpdateValues()
        {
            // Debug.Log($"neighbor idx {tileIndex} updating value");
            bool didValuesUpdate = false;
            for (int i = 0; i < neighbors.Count; i++)
            {
                var n = neighbors[i];
                if (n==null) continue;
                if (n.possibleValues.Count == RuleSet.MaxNumPossibleValues()) continue;
                
                for (int j = possibleValues.Count - 1; j >= 0; j--)
                {
                    TileType p = possibleValues[j];

                    bool isValid = true;

                    foreach (TileType q in n.possibleValues)
                    {
                        if (!RuleSet.IsValidNeighbor(p, q, (Direction)i))
                        {
                            isValid = false;
                            break;
                        }
                    }

                    if (!isValid)
                    {
                        possibleValues.RemoveAt(j);
                        didValuesUpdate = true;
                    }
                }
            }

            if (didValuesUpdate || collapsed)
            {
                Debug.Log($"Tile: {tileIndex} updated, poss values: {possibleValues.Count}.");
                UpdateNeighbors();
            }
            
            
        }

        private void UpdateNeighbors()
        {
            foreach (var n in neighbors)
            {
                if (n == null) continue;
                if (n.collapsed) continue;
                n.UpdateValues();
            }
        }
    }
    
    public class WaveFunctionCollapse:NoiseGenerator
    {
        // rules for what can be next to what. Assume any int can be adjacent to itself and to its neighbours

        private Tile[] tiles;

        private TileType[] values = new[] {TileType.Empty, TileType.Solid};

        private int sizeX, sizeY, sizeZ;

        private int[][] directions =
        {
            new [] {0,0,1},
            new [] {1,0,0},
            new [] {0,0,-1},
            new [] {-1,0,0},
            new [] {0,1,0},
            new [] {0,-1,0},
            
        };

        public override byte[,,] Generate(int x, int y, int z,  Vector3 offset, float cellSize, AdditionalNoiseParameters noiseParams)
        {
            sizeX = x;
            sizeY = y;
            sizeZ = z;
            
            grid = new byte[sizeX, sizeY, sizeZ];
            tiles = new Tile[sizeX*sizeY*sizeZ];

            for (int i = 0; i < tiles.Length; i++)
            {
                tiles[i] = new Tile(values, i);
            }

            for (int i = 0; i < sizeX; i++)
            {
                for (int j = 0; j < sizeY; j++)
                {
                    for (int k = 0; k < sizeZ; k++)
                    {
                        AddNeighbors(i,j,k);
                    }
                }
            }
            
            CollapseLowestEntropyTile();
            
            ConvertTilesToGrid();
            return grid;
        }

        private bool IsValidIndex(int i, int j, int k)
        {
            if (i < 0 || j < 0 || k < 0) return false;
            if (i >= sizeX || j >= sizeY || k >= sizeZ) return false;
            return true;
        }

        private bool IsValidIndex(int i)
        {
            return i >= 0 && i < sizeX * sizeY * sizeZ;
        }

        private int GetRandomTile()
        {
            return Random.Range(0, tiles.Length);
        }

        private void CollapseLowestEntropyTile()
        {
            List<Tile> list = new List<Tile>(tiles);
            list.RemoveAll(a => a.collapsed);
            if (list.Count == 0) return;
            list.Sort((a,b)=>a.entropy.CompareTo(b.entropy));
            var lowestEntropyList = list.FindAll(a => a.entropy == list[0].entropy);
            var tile = lowestEntropyList[Random.Range(0, lowestEntropyList.Count)];
            tile.Collapse();
            CollapseLowestEntropyTile();
        }

        private void AddNeighbors(int i, int j, int k)
        {
            int idx = ThreeToOneDimIndex(i, j, k);
            Tile tile = tiles[idx];
            
            foreach (var d in directions)
            {
                int neighborIdx = ThreeToOneDimIndex(i + d[0], j + d[1], k + d[2]);
                Tile neighborTile = IsValidIndex(neighborIdx) ? tiles[neighborIdx] : null;
                tile.AddNeighbor(neighborTile);
            }
        }

        public void Collapse(int idx)
        {
            tiles[idx].Collapse();
        }

        private void ConvertTilesToGrid()
        {
            for (int i = 0; i < sizeX; i++)
            {
                for (int k = 0; k < sizeZ; k++)
                {
                    for (int j = 0; j < sizeY; j++)
                    {
                        grid[i, j, k] = (byte)tiles[ThreeToOneDimIndex(i,j,k)].value;
                    }
                }
            }
        }
        
        private int ThreeToOneDimIndex(int i, int j, int k)
        {
            return i + k * sizeX + j * sizeX * sizeZ;
        }
        
        private Tuple<int,int,int> OneToThreeDimIndex(int idx)
        {
            int remainder = idx;
            int j = remainder / sizeX*sizeZ;
            remainder %= sizeX * sizeZ;
            int k = remainder / sizeX;
            int i = remainder % sizeX;
            return new Tuple<int, int, int>(i, j, k);

        }
    }
}