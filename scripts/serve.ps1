# Start llama-server for a local GGUF.
# Usage: scripts\serve.ps1 models\Ternary-Bonsai-2-9B-Q2_0.gguf [-Backend vulkan|cpu]
param([Parameter(Mandatory)][string]$Model, [ValidateSet("vulkan","cpu")][string]$Backend = "vulkan",
      [int]$Port = 8080, [int]$Ctx = 4096)
$dir = if ($Backend -eq "vulkan") { "tools\llama-vulkan" } else { "tools\llama.cpp" }
$gpu = if ($Backend -eq "vulkan") { @("-ngl", "99") } else { @("-t", "12") }
& "$dir\llama-server.exe" -m $Model -c $Ctx --port $Port -np 1 -cms 0 -ub 128 --no-webui @gpu
