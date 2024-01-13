#pragma once

#include "VertexArray.h"
#include "IndexBuffer.h"
#include "shaderClass.h"


#define ASSERT(x) if (!(x)) __debugbreak();
#define GlCall(x) GlClearError();\
    x;\
    ASSERT(GLLogCall());


void GClearError();
bool GLLogCall();

class Renderer
{
public:
    void Draw(const VertexArray& va, int count) const;
    void Clear() const;
};
