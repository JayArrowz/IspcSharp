using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace IspcSharp.Generators;

/// <summary>
/// A read-only list with structural (sequence) equality. Incremental-pipeline models must be
/// equatable for the driver to cache them; a plain <see cref="List{T}"/> or
/// <see cref="System.Collections.Immutable.ImmutableArray{T}"/> compares by reference and
/// defeats caching, so models hold their collections through this wrapper instead.
/// </summary>
internal readonly struct EquatableReadOnlyList<T>(IReadOnlyList<T>? collection)
    : IEquatable<EquatableReadOnlyList<T>>, IReadOnlyList<T>
{
    private IReadOnlyList<T> Collection => collection ?? [];

    public T this[int index] => Collection[index];

    public int Count => Collection.Count;

    public bool Equals(EquatableReadOnlyList<T> other)
        => this.SequenceEqual(other);

    public override bool Equals(object? obj)
        => obj is EquatableReadOnlyList<T> other && Equals(other);

    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            foreach (var item in Collection)
                hash = (hash * 31) + (item?.GetHashCode() ?? 0);
            return hash;
        }
    }

    /// <summary>
    /// Index of the first item matching <paramref name="predicate"/>, or -1.
    /// </summary>
    public int FindIndex(Func<T, bool> predicate)
    {
        for (int i = 0; i < Collection.Count; i++)
        {
            if (predicate(Collection[i]))
                return i;
        }

        return -1;
    }

    public IEnumerator<T> GetEnumerator() => Collection.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public static bool operator ==(EquatableReadOnlyList<T> left, EquatableReadOnlyList<T> right)
        => left.Equals(right);

    public static bool operator !=(EquatableReadOnlyList<T> left, EquatableReadOnlyList<T> right)
        => !left.Equals(right);
}
