mkdir -p bin
#clang++ -o bin/libjellyspider-work.so -Isrc/Quartic -Isrc src/jellyspider_work.cc src/Quartic/quartic.cpp -shared
clang++ -o bin/jellyspider src/jellyspider.cc -ldl -DWORK_SO_FILENAME="libjellyspider-work.so" -std=c++20 -lmpfr -lgmp -g -fsanitize=address
