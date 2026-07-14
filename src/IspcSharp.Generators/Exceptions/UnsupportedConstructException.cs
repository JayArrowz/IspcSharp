using System;
using Microsoft.CodeAnalysis;

namespace IspcSharp.Generators.Exceptions;

internal sealed class UnsupportedConstructException(string what, Location? loc = null) : Exception
{
    public string What { get; } = what;
    public Location? Location { get; } = loc;
}