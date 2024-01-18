#define GLM_FORCE_CUDA
#include "kernel.h"
#include <cstdio>
#include <cuda.h>
#include <glm/glm.hpp>
#include <thrust/gather.h>


__global__ void kernel_fish(Fish* fishes, unsigned int* grid_indices, int* grid_cell_start, int* grid_cell_end,
    unsigned int N, BoidsParameters bp, unsigned int grid_size,double mouseX, double mouseY, bool mouse_pressed,
    AnimationVars av, int width, int height, unsigned int* indicies, float cell_width)
{ 
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    int ind = indicies[index];
    if (ind >= N)
    {
        return;
    }

    float xpos_avg = 0.0f, ypos_avg = 0.0f, xvel_avg = 0.0f, yvel_avg = 0.0f, neighboring_boids = 0.0f, close_dx = 0.0f, close_dy = 0.0f;

    Fish fish = fishes[ind]; 


    // find neighbours
    int cell_index = grid_indices[index];
    int row = width / cell_width;
    int neighbour_cells[9];
    for (int i = 0; i < 9; i++)
    {
        if (i < 3)
            neighbour_cells[i] = cell_index + i - 1;
        else if(i < 6)
            neighbour_cells[i] = cell_index + row + i - 3;
        else if(i < 9)
            neighbour_cells[i] = cell_index - row + i - 6;
    }

    // Boids Algoritm
    for (int j = 0; j < 9; ++j)
    {
        int current_cell = neighbour_cells[j];

        if (current_cell < 0 || current_cell >= grid_size)
            continue;

        for (int i = grid_cell_start[current_cell]; i < grid_cell_end[current_cell]; ++i)
        {
            if (i == ind) 
                continue;

            float dx = fish.x - fishes[indicies[i]].x;  
            float dy = fish.y - fishes[indicies[i]].y;

            // Check if fish is in distance
            float distance = glm::sqrt(dx * dx + dy * dy);
            if (glm::abs(dx) < bp.visionRange && glm::abs(dy))
            {

                float distance2 = dx * dx + dy * dy;
                if (distance2 < bp.protectedRange)
                {
                    close_dx += fish.x - fishes[indicies[i]].x;
                    close_dy += fish.y - fishes[indicies[i]].y; 
                }
                else if (distance2 < bp.visionRange * bp.visionRange)
                {
                    xpos_avg += fishes[indicies[i]].x; 
                    ypos_avg += fishes[indicies[i]].y; 
                    xvel_avg += fishes[indicies[i]].vx;
                    yvel_avg += fishes[indicies[i]].vy; 

                    neighboring_boids += 1;
                }

            }
        }
    }

    if (neighboring_boids > 0)
    {
        xpos_avg = xpos_avg / neighboring_boids;
        ypos_avg = ypos_avg / neighboring_boids;
        xvel_avg = xvel_avg / neighboring_boids;
        yvel_avg = yvel_avg / neighboring_boids;

        // Add the centering / matching contributions to velocity
        fish.vx = (fish.vx + (xpos_avg - fish.x) * bp.cohesion + (xvel_avg - fish.vx) * bp.alignment);
        fish.vy = (fish.vy + (ypos_avg - fish.y) * bp.cohesion + (yvel_avg - fish.vy) * bp.alignment);

    }
    // Add the avoidance contribution to velocity
    fish.vx = fish.vx + (close_dx * bp.separation);
    fish.vy = fish.vy + (close_dy * bp.separation);

    // keep fishes in bounds
    if (fish.x < av.margin)
        fish.vx += av.turn_factor;
    if (fish.x > width - av.margin)
        fish.vx -= av.turn_factor;

    if (fish.y < av.margin)
        fish.vy += av.turn_factor;
    if (fish.y > height - av.margin)
        fish.vy -= av.turn_factor;

    if (mouse_pressed)
    {
        double delta_radius = glm::sqrt((mouseX - fish.x) * (mouseX - fish.x) + (mouseY - fish.y) * (mouseY - fish.y));
        double x_diff = mouseX - fish.x;
        double y_diff = mouseY - fish.y;
        if (delta_radius < av.mouse_dist)
        {
            if (x_diff > -av.mouse_dist && x_diff < 0)
                fish.vx += av.turn_factor;
            if (x_diff < av.mouse_dist && x_diff >= 0)
                fish.vx -= av.turn_factor;

            if(y_diff > -av.mouse_dist && y_diff < 0)
                fish.vy += av.turn_factor;
            if (y_diff < av.mouse_dist && y_diff >= 0)
                fish.vy -= av.turn_factor;
        }
    }

    // check if speed is < max_speed, min_speed
    float speed = glm::sqrt(fish.vx * fish.vx + fish.vy * fish.vy);
    if (speed > av.max_speed)
    {
        fish.vx = (fish.vx / speed) * av.max_speed;
        fish.vy = (fish.vy / speed) * av.max_speed;
    }
    if (speed < av.min_speed)
    {
        fish.vx = (fish.vx / speed) * av.min_speed;
        fish.vy = (fish.vy / speed) * av.min_speed;
    }

    fishes[ind].x += fish.vx * bp.speed;
    fishes[ind].y += fish.vy * bp.speed;
    fishes[ind].vx = fish.vx;
    fishes[ind].vy = fish.vy;
}
__global__ void grid_init(Fish* fishes, unsigned int* grid_cells, unsigned int* indices, float cell_width, unsigned int N, int width)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    int col = fishes[index].x / cell_width;
    int row = fishes[index].y / cell_width;

    grid_cells[index] = row * (width / cell_width + 1) + col;
    indices[index] = index;
}

__global__ void grid_start_end(unsigned int* grid_indices, int* grid_cell_start, int* grid_cell_end, unsigned int N)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    if (index == 0)
    {
        grid_cell_start[grid_indices[index]] = 0;
        return;
    }

    if (grid_indices[index] != grid_indices[index - 1])
    {
        grid_cell_end[grid_indices[index - 1]] = index;
        grid_cell_start[grid_indices[index]] = index;
        if (index == N - 1) 
        { 
            grid_cell_end[grid_indices[index]] = index;
        }
    }
}

__global__ void kernel_copy(Fish* fishes, float* vertices, unsigned int N)
{
    const auto i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= N)
    {
        return;
    }
     
    Fish fish = fishes[i];

    vertices[6 * i] = fish.x;;
    vertices[6 * i + 1] = fish.y;

    vertices[6 * i + 2] = fish.x - fish.lenght;
    vertices[6 * i + 3] = fish.y + fish.width / 2;

    vertices[6 * i + 4] = fish.x - fish.lenght;
    vertices[6 * i + 5] = fish.y - fish.width / 2;
}
void CudaFish::memory_alloc(unsigned int N, int width, int height)
{
    AnimationVars av();
    this->width = width;
    this->height = height;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)(&fishes_gpu), N * sizeof(Fish));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)(&grid_indices), N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)(&indices), N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)(&vertices_array_gpu), 6 * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaDeviceSynchronize();
}
void CudaFish::free_memory()
{
    cudaFree(indices);
    cudaFree(grid_indices);
    cudaFree(fishes_gpu);
    cudaFree(vertices_array_gpu);
}
void CudaFish::move_fishes(Fish* fishes, unsigned int N, BoidsParameters bp, double mouseX, double mouseY, bool mouse_pressed)
{
    cudaError_t cudaStatus;
    const dim3 full_blocks_per_grid((N + av.block_size - 1) / av.block_size);
    const dim3 threads_per_block(av.block_size);

    float cell_width = 2 * bp.visionRange;
    unsigned int grid_size = (width / cell_width + 1) * (height / cell_width + 1);
    int* grid_cell_start; int* grid_cell_end;

    cudaStatus = cudaMalloc((void**)(&grid_cell_start), grid_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)(&grid_cell_end), grid_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMemcpy(fishes_gpu, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // // GRID // // 
    grid_init << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu, grid_indices, indices, cell_width, N, width);
    cudaDeviceSynchronize();

    auto thrust_grid_indices = thrust::device_pointer_cast(grid_indices);
    auto thrust_indices = thrust::device_pointer_cast(indices);
    thrust::sort_by_key(thrust_grid_indices, thrust_grid_indices + N, thrust_indices);

    grid_start_end << <full_blocks_per_grid, threads_per_block >> > (grid_indices, grid_cell_start, grid_cell_end, N);
    cudaDeviceSynchronize();

    // Compute fishes
    kernel_fish << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu, grid_indices, grid_cell_start, grid_cell_end,
        N, bp, grid_size, mouseX, mouseY, mouse_pressed, av, width, height, indices, cell_width);
    cudaDeviceSynchronize();

    // Copy data
    cudaStatus = cudaMemcpy(fishes, fishes_gpu, N * sizeof(Fish), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // Free
    cudaFree(grid_cell_start);
    cudaFree(grid_cell_end);
}

void CudaFish::copy_fishes(Fish* fishes, float* vertices_array, unsigned int N)
{
    cudaError_t cudaStatus;

    const dim3 full_blocks_per_grid((N + av.block_size - 1) / av.block_size);
    const dim3 threads_per_block(av.block_size);

    // Copy data to gpu
    cudaStatus = cudaMemcpy(fishes_gpu, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(vertices_array_gpu, vertices_array, 6 * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    kernel_copy << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu, vertices_array_gpu, N);
    cudaDeviceSynchronize();

    // Copy data to CPU
    cudaStatus = cudaMemcpy(vertices_array, vertices_array_gpu, 6 * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

}