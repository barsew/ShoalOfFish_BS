#pragma once

#include <cuda.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include "fish.h"
#include "BoidsParams.h"


// Const variables
typedef struct AnimationVars
{
	unsigned int block_size = 256;
	float max_speed = 3.0f;
	float min_speed = 2.0f;
	float margin = 50.0f;
	float turn_factor = 2.f;
	float mouse_dist = 100.0f;
};


class CudaFish
{
public:
	void memory_alloc(unsigned int N, int width, int height);
	void free_memory();
	void move_fishes(Fish* fishes, unsigned int N, BoidsParameters bp, double mouseX, double mouseY, bool mouse_pressed);
	void copy_fishes(Fish* fishes, float* vertices_array, unsigned int N);
private:
	int width;
	int height;
	AnimationVars av;
	Fish* fishes_gpu;
	unsigned int* indices;
	unsigned int* grid_indices;
	float* vertices_array_gpu;
};

