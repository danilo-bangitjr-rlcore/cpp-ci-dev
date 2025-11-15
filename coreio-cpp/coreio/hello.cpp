#include "hello.h"

std::string hello_world_message() { 
    return std::string(new char[100]{"Hello, World!"});
}
