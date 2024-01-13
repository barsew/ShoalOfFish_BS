#define GLM_FORCE_CUDA
#include "kernel.h"
#include <cstdio>
#include <cuda.h>
#include <glm/glm.hpp>
#include <thrust/gather.h>

// Const variables
constexpr unsigned int block_size = 256;
constexpr float max_speed = 10.0f;
constexpr float min_speed = 2.0f;
constexpr float turn_factor = 2.f;
constexpr float height = 900.f;
constexpr float width = 1600.f;
constexpr float margin = 50;


// Assign helpful variables
//glm::vec2* velocity_buffer = nullptr;
//Fish* fishes_gpu = nullptr;
//Fish* fishes_gpu_sorted = nullptr;
//unsigned int* indices = nullptr;
//unsigned int* grid_cell_indices = nullptr;
//float* vertices_array_gpu = nullptr;

// Compute velocity kernel function
__global__ void compute_vel(Fish* fishes, glm::vec2* vel2, unsigned int* grid_cell_indices, int* grid_cell_start, int* grid_cell_end,
    unsigned int N, float visualRange, float protectedRange, float cohesion_scale, float separation_scale, float alignment_scale, unsigned int grid_size,
    double mouseX, double mouseY, bool mouse_pressed, bool group_by_species)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    float xpos_avg = 0.0f, ypos_avg = 0.0f, xvel_avg = 0.0f, yvel_avg = 0.0f, neighboring_boids = 0.0f, close_dx = 0.0f, close_dy = 0.0f;

    Fish fish = fishes[index];

    // Find neighbours cells
    int cell_index = grid_cell_indices[index];
    int row_cells = width / (2 * visualRange) + 1;
    int neighbour_cells[] = {cell_index, cell_index + 1, cell_index - 1, cell_index + row_cells, cell_index - row_cells, cell_index - row_cells - 1,
                    cell_index - row_cells + 1, cell_index + row_cells - 1, cell_index + row_cells + 1 };

    // Go through neighbour cells
    for (int j = 0; j < 9; ++j)
    {
        int current_cell = neighbour_cells[j];

        if (current_cell < 0 || current_cell >= grid_size)
            continue;

        // Iterate through fishes from neighbour cell
        for (int i = grid_cell_start[current_cell]; i < grid_cell_end[current_cell]; ++i)
        {
            if (i == index)
                continue;

            float dx = fish.x - fishes[i].x;
            float dy = fish.y - fishes[i].y;

            // Check if fish is in distance
            float distance = glm::sqrt(dx * dx + dy * dy);
            if (glm::abs(dx) < visualRange && glm::abs(dy))
            {

                float distance2 = dx * dx + dy * dy;
                if (distance2 < protectedRange)
                {
                    close_dx += fish.x - fishes[i].x;
                    close_dy += fish.y - fishes[i].y;
                }
                else if (distance2 < visualRange * visualRange)
                {
                    xpos_avg += fishes[i].x;
                    ypos_avg += fishes[i].y;
                    xvel_avg += fishes[i].vx;
                    yvel_avg += fishes[i].vy;

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
        fish.vx = (fish.vx + (xpos_avg - fish.x) * cohesion_scale + (xvel_avg - fish.vx) * alignment_scale);
        fish.vy = (fish.vy + (ypos_avg - fish.y) * cohesion_scale + (yvel_avg - fish.vy) * alignment_scale);

    }
    // Add the avoidance contribution to velocity
    fish.vx = fish.vx + (close_dx * separation_scale);
    fish.vy = fish.vy + (close_dy * separation_scale);

    // keep fishes in bounds
    if (fish.x < margin)
        fish.vx += turn_factor;
    if (fish.x > width - margin)
        fish.vx -= turn_factor;

    if (fish.y < margin)
        fish.vy += turn_factor;
    if (fish.y > height - margin)
        fish.vy -= turn_factor;

    // avoid coursor if mouse pressed
    if (mouse_pressed)
    {
        double x_diff = mouseX - fish.x;
        double y_diff = mouseY - fish.y;
        if (x_diff < margin && x_diff > -margin && y_diff < margin && y_diff > -margin)
        {
            if (x_diff > -margin && x_diff < 0)
                fish.vx += turn_factor;
            if (x_diff < margin && x_diff >= 0)
                fish.vx -= turn_factor;

            if(y_diff > -margin && y_diff < 0)
                fish.vy += turn_factor;
            if (y_diff < margin && y_diff >= 0)
                fish.vy -= turn_factor;
        }
    }

    // check if speed is < max_speed
    float speed = glm::sqrt(fish.vx * fish.vx + fish.vy * fish.vy);
    if (speed > max_speed)
    {
        fish.vx = (fish.vx / speed) * max_speed;
        fish.vy = (fish.vy / speed) * max_speed;
    }
    if (speed < min_speed)
    {
        fish.vx = (fish.vx / speed) * min_speed;
        fish.vy = (fish.vy / speed) * min_speed;
    }

    // update velocities
    vel2[index].x = fish.vx;
    vel2[index].y = fish.vy;
}

// Update pos and velocity kernel function
__global__ void update_pos_vel(Fish* fishes, glm::vec2* vel, unsigned int N, float speed_scale)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    fishes[index].x += vel[index].x * speed_scale;
    fishes[index].y += vel[index].y * speed_scale;

    if (fishes[index].x < 0)
        fishes[index].x = 0;
    if (fishes[index].x > width)
        fishes[index].x = width;
    if (fishes[index].y < 0)
        fishes[index].y = 0;
    if (fishes[index].y > height)
        fishes[index].y = height;

    fishes[index].vx = vel[index].x;
    fishes[index].vy = vel[index].y;
}

// Assign grid cell to fish
__global__ void assign_grid_cell(Fish* fishes, unsigned int* grid_cells, unsigned int* indices, float cell_width, unsigned int N)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    float x = fishes[index].x;
    float y = fishes[index].y;

    int x_size = width / cell_width + 1;

    int x_cell = x / cell_width;
    int y_cell = y / cell_width;

    grid_cells[index] = y_cell * x_size + x_cell;
    indices[index] = index;
}

// Compute start and end indices of cell
__global__ void compute_start_end_cell(unsigned int* grid_cell_indices, int* grid_cell_start, int* grid_cell_end, unsigned int N)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    unsigned int grid_cell_id = grid_cell_indices[index];

    if (index == 0)
    {
        grid_cell_start[grid_cell_id] = 0;
        return;
    }
    unsigned int prev_grid_cell_id = grid_cell_indices[index - 1];
    if (grid_cell_id != prev_grid_cell_id)
    {
        grid_cell_end[prev_grid_cell_id] = index;
        grid_cell_start[grid_cell_id] = index;
        if (index == N - 1) 
        { 
            grid_cell_end[grid_cell_id] = index;
        }
    }
}

// Copy new vertices to array
__global__ void copy_fishes_kernel(Fish* fishes, float* vertices, unsigned int N)
{
    const auto i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= N) { return; }

    Fish fish = fishes[i];

    vertices[i * 3 * 5] = fish.x;;
    vertices[i * 3 * 5 + 1] = fish.y;

    vertices[i * 3 * 5 + 2] = 0; vertices[i * 3 * 5 + 3] = 0; vertices[i * 3 * 5 + 4] = 0;

    vertices[i * 3 * 5 + 5] = fish.x - fish.lenght;
    vertices[i * 3 * 5 + 6] = fish.y + fish.width / 2;

    vertices[i * 3 * 5 + 7] = 0; vertices[i * 3 * 5 + 8] = 0; vertices[i * 3 * 5 + 9] = 0;

    vertices[i * 3 * 5 + 10] = fish.x - fish.lenght;
    vertices[i * 3 * 5 + 11] = fish.y - fish.width / 2;

    vertices[i * 3 * 5 + 12] = 0; vertices[i * 3 * 5 + 13] = 0; vertices[i * 3 * 5 + 14] = 0;
}

// Allocate needed memory
void CudaFish::initialize_simulation(unsigned int N)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
    }

    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&velocity_buffer), N * sizeof(glm::vec2));
    if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&grid_cell_indices), N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&indices), N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&fishes_gpu), N * sizeof(Fish));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&fishes_gpu_sorted), N * sizeof(Fish));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&vertices_array_gpu), 15 * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaDeviceSynchronize();
}
// Deallocate memory
void CudaFish::end_simulation()
{
    cudaFree(velocity_buffer);
    cudaFree(indices);
    cudaFree(grid_cell_indices);
    cudaFree(fishes_gpu);
    cudaFree(fishes_gpu_sorted);
    cudaFree(vertices_array_gpu);
}
// Function to upddate fish pos and vel
void CudaFish::update_fishes(Fish* fishes, unsigned int N, float vr, float md, float r1, float r2, float r3, float speed_scale, double mouseX, double mouseY, bool mouse_pressed, bool group_by_species)
{
    cudaError_t cudaStatus;
    
    const dim3 full_blocks_per_grid((N + block_size - 1) / block_size);
    const dim3 threads_per_block(block_size);

    float cell_width = 2 * vr;
    unsigned int grid_size = (width / cell_width + 1) * (height / cell_width + 1);

    // Allocate memory for start and end indices
    int* grid_cell_start;
    int* grid_cell_end;

    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&grid_cell_start), grid_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&grid_cell_end), grid_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy data to gpu
    cudaStatus = cudaMemcpy(fishes_gpu, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // Asign grid cell to every fish
    assign_grid_cell << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu, grid_cell_indices, indices, cell_width, N);
    cudaDeviceSynchronize();


    // Cast arrays to perform thrust operations
    auto thrust_gci = thrust::device_pointer_cast(grid_cell_indices);
    auto thrust_i = thrust::device_pointer_cast(indices);
    auto thrust_f = thrust::device_pointer_cast(fishes_gpu);
    auto thrust_fs = thrust::device_pointer_cast(fishes_gpu_sorted);

    // Sort fishes indicies by grid cell
    thrust::sort_by_key(thrust_gci, thrust_gci + N, thrust_i);

    // Compute start and end indices of grid cell
    compute_start_end_cell << <full_blocks_per_grid, threads_per_block >> > (grid_cell_indices, grid_cell_start, grid_cell_end, N);
    cudaDeviceSynchronize();

    // Sort fish pos and vel by indices
    thrust::gather(thrust_i, thrust_i + N, thrust_f, thrust_fs);


    // Update velocity
    compute_vel << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu_sorted, velocity_buffer, grid_cell_indices, grid_cell_start, grid_cell_end,
        N, vr, md, r1, r2, r3, grid_size,
        mouseX, mouseY, mouse_pressed, group_by_species);
    cudaDeviceSynchronize();


    // Update position
    update_pos_vel << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu_sorted, velocity_buffer, N, speed_scale);
    cudaDeviceSynchronize();
    

    // Copy data to CPU
    cudaStatus = cudaMemcpy(fishes, fishes_gpu_sorted, N * sizeof(Fish), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // Free memory
    cudaFree(grid_cell_start);
    cudaFree(grid_cell_end);
}
// Copy fishes to VBO
void CudaFish::copy_fishes(Fish* fishes, float* vertices_array, unsigned int N)
{
    cudaError_t cudaStatus;

    const dim3 full_blocks_per_grid((N + block_size - 1) / block_size);
    const dim3 threads_per_block(block_size);

    // Copy data to gpu
    cudaStatus = cudaMemcpy(fishes_gpu, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(vertices_array_gpu, vertices_array, 15 * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    copy_fishes_kernel << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu, vertices_array_gpu, N);
    cudaDeviceSynchronize();

    // Copy data to CPU
    cudaStatus = cudaMemcpy(vertices_array, vertices_array_gpu, 15 * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

}