project "Voxelion-App"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++20"
   targetdir "Binaries/%{cfg.buildcfg}"
   staticruntime "off"

   files { "Source/**.h", "Source/**.cpp" }

   prebuildcommands {
      "glslc %[%{!prj.location}/Shaders/shader.vert] -o %[%{!prj.location}/ShadersCompiled/vert.spv]",
      "glslc %[%{!prj.location}/Shaders/shader.frag] -o %[%{!prj.location}/ShadersCompiled/frag.spv]",
   }

   includedirs
   {
      "Source",

	  -- Include Core
	  "../Voxelion-Core/Source",
      "../Vendor/glm"
   }

   links
   {
      "Voxelion-Core",
      "GLFW",
      "%{Library.Vulkan}",
   }

   targetdir ("../Binaries/" .. OutputDir .. "/%{prj.name}")
   objdir ("../Binaries/Intermediates/" .. OutputDir .. "/%{prj.name}")

   filter "system:windows"
       systemversion "latest"
       defines { "WINDOWS" }

   filter "configurations:Debug"
       defines { "DEBUG" }
       runtime "Debug"
       symbols "On"

   filter "configurations:Release"
       defines { "RELEASE" }
       runtime "Release"
       optimize "On"
       symbols "On"

   filter "configurations:Dist"
       defines { "DIST" }
       runtime "Release"
       optimize "On"
       symbols "Off"