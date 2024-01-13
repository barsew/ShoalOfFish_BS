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
	float max_speed = 10.0f;
	float min_speed = 2.0f;
	float margin = 50;
	float turn_factor = 2.f;
};


class CudaFish
{
public:
	void initialize_simulation(unsigned int N, int width, int height);
	void end_simulation();
	void update_fishes(Fish* fishes, unsigned int N, BoidsParameters bp, double mouseX, double mouseY, bool mouse_pressed);
	void copy_fishes(Fish* fishes, float* vertices_array, unsigned int N);
private:
	int width;
	int height;
	AnimationVars av;
	glm::vec2* velocity_buffer;
	Fish* fishes_gpu;
	Fish* fishes_gpu_sorted;
	unsigned int* indices;
	unsigned int* grid_cell_indices;
	float* vertices_array_gpu;
};

