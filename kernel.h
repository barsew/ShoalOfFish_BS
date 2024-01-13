#pragma once

#include <cuda.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include "fish.h"

//namespace CudaFunctions
//{
//	void initialize_simulation(unsigned int N);
//	void end_simulation();
//	void update_fishes(Fish* fishes, unsigned int N, float vr, float md, float r1, float r2, float r3, float dt, double mouseX, double mouseY, bool mouse_pressed, bool group_by_species);
//	void copy_fishes(Fish* fishes, float* vertices_array, unsigned int N);
//}

class CudaFish
{
public:
	void initialize_simulation(unsigned int N);
	void end_simulation();
	void update_fishes(Fish* fishes, unsigned int N, float vr, float md, float r1, float r2, float r3, float dt, double mouseX, double mouseY, bool mouse_pressed, bool group_by_species);
	void copy_fishes(Fish* fishes, float* vertices_array, unsigned int N);
private:
	glm::vec2* velocity_buffer;
	Fish* fishes_gpu;
	Fish* fishes_gpu_sorted;
	unsigned int* indices;
	unsigned int* grid_cell_indices;
	float* vertices_array_gpu;
};
