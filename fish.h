#pragma once

#include<GLFW/glfw3.h>
#include <glm/glm.hpp>


struct Fish
{
	float x;
	float y;
	float vx;
	float vy;

	float lenght = 15;	// fish params
	float width = 5;

	Fish(float x, float y)
	{
		this->x = x;
		this->y = y;
		vx = vy = 0;
	}


	Fish()
	{
		x = 0;
		y = 0;
		vx = 0;
		vy = 0;
	}
};