#include "Renderer.h"
#include <iostream>


void GClearError()
{
    while (glGetError() != GL_NO_ERROR);
}

bool GLLogCall()
{
    while (GLenum error = glGetError())
    {
        std::cout << "[OpenGL Error] (" << error << ")" << std::endl;
        return false;
    }

    return true;
}

void Renderer::Draw(const VertexArray& va, int count) const
{
    va.Bind();
    glDrawArrays(GL_TRIANGLES, 0, count);
    va.Unbind();
}

void Renderer::Clear() const
{
    glClear(GL_COLOR_BUFFER_BIT);
}