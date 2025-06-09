-- VoxelionExtra.lua

VULKAN_SDK = os.getenv("VULKAN_SDK")

IncludeDir = {}
IncludeDir["VulkanSDK"] = "/usr/include"
IncludeDir["glm"] = "./Vendor/glm"
IncludeDir["stb"] = "./Vendor/stb"

LibraryDir = {}
LibraryDir["VulkanSDK"] = "/usr/share"

Library = {}
Library["Vulkan"] = "%{LibraryDir.VulkanSDK}/vulkan"

-- filter "system:windows"
--    IncludeDir["VulkanSDK"] = "%{VULKAN_SDK}/Include"
--    LibraryDir["VulkanSDK"] = "%{VULKAN_SDK}/Lib"
--    Library["Vulkan"] = "%{LibraryDir.VulkanSDK}/vulkan-1.lib"

group "Dependencies"
   include "Vendor/glfw"
group ""
