#include<iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include "kernel.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "shaderClass.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "kernel.h"
#include "fish.h"
#include <ctime>
#include <imgui_impl_opengl3.h>
#include <chrono>
#include "VertexArray.h"
#include "Renderer.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1000

// number od fishes
#define N 10000

bool init_window();
void init_fishes();
void cursor_callback(GLFWwindow* window, double x, double y);
void mouse_callback(GLFWwindow* window, int button, int action, int mods);


GLFWwindow* window = nullptr;
Shader shader;

// Fishes info
Fish fishes[N];
float vertices[6 * N];


// Mouse avoid
double mouseX = 0;
double mouseY = 0;
bool mouse_pressed = false;


// Main function
int main()
{
	if (init_window() == -1)
		return -1;

	init_fishes();
	CudaFish cf;
	cf.memory_alloc(N, WINDOW_WIDTH, WINDOW_HEIGHT);

	// Boids params
	BoidsParameters bp;

	for (int i = 0; i < N; ++i)
	{
		Fish fish = fishes[i];

		vertices[6 * i] = fish.x;;
		vertices[6 * i + 1] = fish.y;

		vertices[6 * i + 2] = fish.x - fish.lenght;
		vertices[6 * i + 3] = fish.y + fish.width / 2;

		vertices[6 * i + 4] = fish.x - fish.lenght;
		vertices[6 * i + 5] = fish.y - fish.width / 2;
	}

	VertexArray va;
	VertexBuffer vb(vertices, sizeof(vertices));

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	Renderer renderer;

	while (!glfwWindowShouldClose(window))
	{
		auto start = std::chrono::high_resolution_clock::now();

		// Take care of all GLFW events
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Clean the back buffer and assign the new color to it
		glClearColor(0.3f, 0.3f, 0.6f, 1.0f);
		renderer.Clear();


		cf.move_fishes(fishes, N, bp, mouseX, mouseY, mouse_pressed);

		float vertices[6 * N];
		cf.copy_fishes(fishes, vertices, N);

		va.Bind();
		vb.Bind();
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
		vb.Unbind();
		va.Unbind();

		renderer.Draw(va, 3 * N);

		ImGui::Begin("Set properties");
		ImGui::Text("Liczba rybek %d", N);
		ImGui::SliderFloat("Visual range of fish", &bp.visionRange, 5.0f, 100.0f);
		ImGui::SliderFloat("Protected range", &bp.protectedRange, 0.0f, 100.0f);
		ImGui::SliderFloat("Cohesion rule scale", &bp.cohesion, 0.0f, 0.1f);
		ImGui::SliderFloat("Separation rule scale", &bp.separation, 0.0f, 0.1f);
		ImGui::SliderFloat("Alignment rule scale", &bp.alignment, 0.0f, 0.1f);
		ImGui::SliderFloat("Speed scale", &bp.speed, 0.1f, 0.5f);

		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// Swap the back buffer with the front buffer
		//glfwSwapBuffers(window);
		glFlush();

		// fps
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		float fps = 1000000.0f / static_cast<float>(duration);
		std::cout << "FPS: " << fps << std::endl;
	}

	// Clear memory
	shader.unbind();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	cf.free_memory();

	// Delete window before ending the program
	glfwDestroyWindow(window);
	// Terminate GLFW before ending the program
	glfwTerminate();

	return 0;
}

void mouse_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		double x;
		double y;
		glfwGetCursorPos(window, &x, &y);
		mouseX = x;
		mouseY = WINDOW_HEIGHT - y;
		mouse_pressed = true;
	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		mouse_pressed = false;
	}
}
void cursor_callback(GLFWwindow* window, double x, double y)
{
	if (mouse_pressed)
	{
		mouseX = x;
		mouseY = WINDOW_HEIGHT - y;
	}
}
void init_fishes()
{
	for (int i = 0; i < N; ++i)
	{
		float X = (static_cast<float>(rand() % static_cast<int>(WINDOW_WIDTH)));
		float Y = (static_cast<float>(rand() % static_cast<int>(WINDOW_HEIGHT)));

		int x_vel = rand() % 10 + 1;
		int y_vel = rand() % 10 + 1;
		int x_rand = rand() % 2;
		int y_rand = rand() % 2;

		fishes[i] = Fish(X, Y);

		if (x_rand == 0)
			fishes[i].vx = x_vel;
		else
			fishes[i].vx = -x_vel;

		if (y_rand == 0)
			fishes[i].vy = y_vel;
		else
			fishes[i].vy = -y_vel;

	}
}
bool init_window()
{
	srand(time(NULL));

	// Initialize GLFW
	if (!glfwInit())
	{
		std::cout << "GLFW init failed";
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);

	// Create a GLFWwindow object
	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Aquarium", NULL, NULL);

	// Error check if the window fails to create
	if (window == NULL)
	{
		std::cout << "Failed to create window";
		glfwTerminate();
		return false;
	}

	glfwSetMouseButtonCallback(window, mouse_callback);
	glfwSetCursorPosCallback(window, cursor_callback);

	// Introduce the window into the current context
	glfwMakeContextCurrent(window);

	gladLoadGL();

	// Initialize shaders
	shader = Shader("default");
	shader.bind();
	glm::mat4 proj = glm::ortho(0.f, static_cast<float>(WINDOW_WIDTH), 0.f, static_cast<float>(WINDOW_HEIGHT), -1.f, 1.f);
	shader.setUniformMat4fv("projection", proj);

	// Initialize ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	return true;
}
