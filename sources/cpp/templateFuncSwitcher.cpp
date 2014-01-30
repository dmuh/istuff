#include <iostream>

template<typename T>
class FunctionSwitcher
{
public:
    template<typename FloatFuncType, typename DoubleFuncType>
    static FloatFuncType getFunc(FloatFuncType f1, DoubleFuncType f2)
    {
        return f1;
    }
};

template<>
class FunctionSwitcher<double>
{
public:
    template<typename FloatFuncType, typename DoubleFuncType>
    static DoubleFuncType getFunc(FloatFuncType f1, DoubleFuncType f2)
    {
        return f2;
    }
};

// using example
double someDoubleLogic(float f, float b)
{
    return f + b;
}
float someFloatLogic(int f)
{
    return f;
}

//in case if we have different arguments count
#define SmartFunc(a, ...) FunctionSwitcher<a>::getFunc(someFloatLogic, someDoubleLogic)(__VA_ARGS__)

//in case if we have the same arguments count
double someAnotherLogic(float f)
{
    return f * f;
}
template<typename T>
T smartFunc(T var)
{
    return FunctionSwitcher<T>::getFunc(someFloatLogic, someAnotherLogic)(var);
}


int main(int argc, const char** argv)
{
    std::cout <<"Define float: " << SmartFunc(float, 11)<< std::endl;
    std::cout <<"Define double: " << SmartFunc(double, 11, 2)<< std::endl;

    std::cout << "Template float: " << smartFunc(11.f) << std::endl;
    std::cout << "Template another double: " << smartFunc(11.0) << std::endl;
    return 0;
}
