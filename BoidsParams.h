
typedef struct BoidsParameters
{

    float cohesion = 0.001f;
    float separation = 0.05f;
    float alignment = 0.05f;

    float speed = 0.1f;
    float visionRange = 35.f;
    float protectedRange = 20.f;

    BoidsParameters(float cohesion = 0.001f, float separation = 0.05f, float aligment = 0.05f, float speed = 0.1f, float visionRange = 35.0f, float protectedRange = 20.0f)
    {
        this->cohesion = cohesion;
        this->separation = separation;
        this->alignment = aligment;
        this->speed = speed;
        this->visionRange = visionRange;
        this->protectedRange = protectedRange;
    }
};