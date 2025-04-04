#pragma once

#include <memory>

namespace Core {

	void PrintHelloWorld();

	namespace VoxelType {
		enum Type {
			Air = 0,
			Dirt = 1,
			Stone = 2,
		};
	}

	namespace LayerType {
		enum Type {
			// Type data for all blocks
			Full = 0,
			// full layer of same blocks
			SingleType = 1,
			// [offsets for rows] + {block * count}
			Indexed = 2,
		};
	}

	class Layer {
	public:
		LayerType::Type type;
		void *data;
	};


	class Chunk {
		// Contains layers of blocks
		// Every chunk is 32x32x64 so we
		//   can use 16 bit int to index into chunk
	};
}