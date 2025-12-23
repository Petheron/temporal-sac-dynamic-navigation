using System;

[Flags]
public enum MutatorType
{
    None = 0,
    Water = 1 << 0,
    Ice = 1 << 1,
    Slow = 1 << 2, 
    LowGrav = 1 << 3,
    HighGrav = 1 << 4,
    Space = 1 << 5
}

