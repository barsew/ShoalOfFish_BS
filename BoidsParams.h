
typedef struct BoidsParameters
{
    float cohesion = 0.001f;
    float separation = 0.05f;
    float alignment = 0.05f;

    float speed = 0.1f;
    float visionRange = 30.f;
    float protectedRange = 20.f;
};