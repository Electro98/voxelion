-- Build.lua

workspace "Voxelion"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "App"

   -- Workspace-wide build options for MSVC
   filter "system:windows"
      buildoptions { "/EHsc", "/Zc:preprocessor", "/Zc:__cplusplus" }

OutputDir = "%{cfg.system}-%{cfg.architecture}/%{cfg.buildcfg}"

include "VoxelionExternal.lua"

group "Voxelion-Core"
	include "Voxelion-Core/Build-Core.lua"
group ""

include "Voxelion-App/Build-App.lua"