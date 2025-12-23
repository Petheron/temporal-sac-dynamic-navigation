using System;

[Flags]
public enum VisionTag
{
    Solid= 1<<0,
    Danger=1<<1,
    Independent=1<<2,
    Affector=1<<3,
    Affected=1<<4,
    Agent=1<<5,
    DeathField=1<<6
}