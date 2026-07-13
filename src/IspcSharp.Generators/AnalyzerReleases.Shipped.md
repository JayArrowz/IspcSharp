## Release 1.0

### New Rules

Rule ID | Category | Severity | Notes                                
--------|----------|----------|--------------------
ISPC001 | IspcSharp | Error   | Containing type must be partial              
ISPC002 | IspcSharp | Error   | Unsupported [Spmd] method shape              
ISPC003 | IspcSharp | Error   | Construct not vectorizable                   
ISPC004 | IspcSharp | Error   | Unsupported parameter type                   
ISPC005 | IspcSharp | Info    | Parallel variant skipped                     
ISPC100 | IspcSharp.Performance | Warning  | Array-of-Structs access in SPMD kernel
ISPC101 | IspcSharp.Performance | Warning  | Gather (non-contiguous load) in SPMD kernel  
ISPC102 | IspcSharp.Performance | Warning  | Scatter (non-contiguous store) in SPMD kernel
ISPC103 | IspcSharp.Performance | Warning  | Per-lane integer divide in SPMD kernel       
ISPC104 | IspcSharp.Performance | Warning  | Double↔integer conversion in SPMD kernel     
