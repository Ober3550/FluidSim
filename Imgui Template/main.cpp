#include "imgui.h"
#include "imgui-SFML.h"
#include "zlib.h"

#include <SFML/Graphics.hpp>
#include <iostream>

sf::Texture square;

struct SmallPos {
    int8_t x;
    int8_t y;
};

std::vector<SmallPos> orth_dirs = {
    { 0, -1},
    { 0,  1},
    {-1,  0},
    { 1,  0},
};

std::vector<SmallPos> diag_dirs = {
    {-1, -1},
    { 1,  1},
    {-1,  1},
    { 1, -1},
};

std::vector<SmallPos> both_dirs = {
    { 0, -1},
    { 0,  1},
    {-1,  0},
    { 1,  0},
    {-1, -1},
    { 1,  1},
    {-1,  1},
    { 1, -1},
};

#define N 50
#define BOARD_WIDTH 50
#define BOARD_HEIGHT 50
#define HALF_WIDTH 25
#define HALF_HEIGHT 25
#define SCALE 0.25f
#define GRID_SIZE 64.f
#define ITER 4

int IX(int x, int y) {
    assert((x >= 0 && x < N));
    assert((y >= 0 && y < N));
    return x + y * N;
}

class FluidGrid {
public:
    float dt;
    float diff;
    float visc;
    std::vector<float> s;
    std::vector<float> den;
    std::vector<float> Vx;
    std::vector<float> Vy;
    std::vector<float> Vx0;
    std::vector<float> Vy0;
    FluidGrid(float dt, float diffusion, float viscosity) {
        this->dt = dt;
        this->diff = diffusion;
        this->visc = viscosity;
        for (int i = 0; i < N * N; i++) {
            this->s.emplace_back(0.f);
            this->den.emplace_back(0.f);
            this->Vx.emplace_back(0.f);
            this->Vy.emplace_back(0.f);
            this->Vx0.emplace_back(0.f);
            this->Vy0.emplace_back(0.f);
        }
    }
    void AddDensity(int x, int y, float amount) {
        int index = IX(x, y);
        this->den[index] += amount;
    }
    void AddVelocity(int x, int y, float amountX, float amountY) {
        int index = IX(x, y);
        this->Vx[index] += amountX;
        this->Vy[index] += amountY;
    }
    void set_bnd(int b, std::vector<float>& d)
    {
        // Edge Cases
        for (int i = 1; i < N - 1; i++) {
            d[IX(i, 0)]     = b == 2 ? -d[IX(i    , 1)] : d[IX(i, 1    )];
            d[IX(i, N - 1)] = b == 2 ? -d[IX(i, N - 2)] : d[IX(i, N - 2)];
        }
        for (int j = 1; j < N - 1; j++) {
            d[IX(0, j)]     = b == 1 ? -d[IX(1    , j)] : d[IX(1    , j)];
            d[IX(N - 1, j)] = b == 1 ? -d[IX(N - 2, j)] : d[IX(N - 2, j)];
        }
        // Corner Cases
        d[IX(0  , 0  )] = 0.5f * (d[IX(1  , 0  )] + d[IX(0  , 1  )]);
        d[IX(0  , N-1)] = 0.5f * (d[IX(1  , N-1)] + d[IX(0  , N-2)]);
        d[IX(N-1, 0  )] = 0.5f * (d[IX(1  , 0  )] + d[IX(0  , 1  )]);
        d[IX(N-1, N-1)] = 0.5f * (d[IX(N-2, N-1)] + d[IX(N-1, N-2)]);
    }
    // Can project current onto previous and previous onto current
    void project(std::vector<float>& velX, std::vector<float>& velY, std::vector<float>& p, std::vector<float>& div)
    {
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                div[IX(i, j)] = -0.5f * (
                      velX[IX(i + 1, j)]
                    - velX[IX(i - 1, j)]
                    + velY[IX(i, j + 1)]
                    - velY[IX(i, j - 1)]
                    ) / N;
                p[IX(i, j)] = 0;
            }
        }
        set_bnd(0, div);
        set_bnd(0, p);
        lin_solve(0, p, div, 1, 6);
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                velX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N;
                velY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N;
            }
        }
        set_bnd(1, velX);
        set_bnd(2, velY);
    }
    void lin_solve(int b, std::vector<float>& d, std::vector<float>& d0, float a, float c) {
        float cRecip = 1.0 / c;
        for (int m = 0; m < ITER; m++) {
            for (int j = 1; j < N - 1; j++) {
                for (int i = 1; i < N - 1; i++) {
                    d[IX(i, j)] = cRecip * (d0[IX(i, j)] +
                        a * (d[IX(i + 1, j    )] +
                             d[IX(i - 1, j    )] +
                             d[IX(i    , j + 1)] +
                             d[IX(i    , j - 1)]));
                }
            }
            set_bnd(b, d);
        }
    }
    void diffuse(int b, std::vector<float>& d, std::vector<float>& d0, float diff, float dt) {
        float a = dt * diff * (N - 2) * (N - 2);
        lin_solve(b, d, d0, a, 1 + 6 * a);
    }
    void advect(int b, std::vector<float>& d, std::vector<float>& d0, std::vector<float>& velX, std::vector<float>& velY, float dt)
    {
        float i0, i1, j0, j1;

        float dtx = dt * (N - 2);
        float dty = dt * (N - 2);

        float s0, s1, t0, t1;
        float tmp1, tmp2, x, y;

        float Nfloat = N;
        float ifloat, jfloat;

        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                tmp1 = dtx * velX[IX(i, j)];
                tmp2 = dty * velY[IX(i, j)];
                x = float(i) - tmp1;
                y = float(j) - tmp2;

                if (x < 0.5f) x = 0.5f;
                if (x > Nfloat + 0.5f) x = Nfloat + 0.5f;
                i0 = floorf(x);
                i1 = i0 + 1.0f;
                if (y < 0.5f) y = 0.5f;
                if (y > Nfloat + 0.5f) y = Nfloat + 0.5f;
                j0 = floorf(y);
                j1 = j0 + 1.0f;

                s1 = x - i0;
                s0 = 1.0f - s1;
                t1 = y - j0;
                t0 = 1.0f - t1;

                int i0i = i0;
                int i1i = i1;
                int j0i = j0;
                int j1i = j1;

                d[IX(i, j)] =
                    s0 * (t0 * d0[IX(i0i, j0i)]
                        + t1 * d0[IX(i0i, j1i)])
                  + s1 * (t0 * d0[IX(i1i, j0i)]
                        + t1 * d0[IX(i1i, j1i)]);
            }
        }
        set_bnd(b, d);
    }
    void step() {
        diffuse(1, Vx0, Vx, visc, dt);
        diffuse(2, Vy0, Vy, visc, dt);

        project(Vx0, Vy0, Vx, Vy);

        advect(1, Vx, Vx0, Vx0, Vy0, dt);
        advect(2, Vy, Vy0, Vx0, Vy0, dt);
        
        project(Vx, Vy, Vx0, Vy0);
        
        diffuse(0, s, den, diff, dt);
        advect(0, den, s, Vx, Vy, dt);
    }
};

std::vector<sf::Sprite> sprites;
void drawState(const FluidGrid& simulation) {
    for (int j = 0; j < BOARD_HEIGHT; j++) {
        for (int i = 0; i < BOARD_WIDTH; i++) {
            sf::Sprite sprite;
            sf::Vector3f density = sf::Vector3f(1,1,1) * simulation.den[IX(i, j)];
            sprite.setColor(sf::Color(density.x, density.y, density.z));
            sprite.setTexture(square);
            sprite.setScale(SCALE, SCALE);
            sprite.setPosition((i - HALF_WIDTH) * GRID_SIZE * SCALE, (j - HALF_HEIGHT) * GRID_SIZE * SCALE);
            sprites.emplace_back(sprite);
        }
    }
}

int main()
{
    sf::RenderWindow window(sf::VideoMode(1000, 1000), "Fluid Sim");
    window.setFramerateLimit(60);
    ImGui::SFML::Init(window);

    square.loadFromFile("square.png");
    FluidGrid simulation = FluidGrid(0.1,0,0);
    sf::View centreView;
    sf::Vector2u size = window.getSize();
    centreView.setSize(sf::Vector2f(size.x, size.y));
    centreView.setCenter(0, 0);
    sf::Vector2f mousePos = { 0.f,0.f };
    SmallPos selectPos = { -1, -1 };
    SmallPos prevBoardPos = { 0, 0 };
    SmallPos boardPos = { 0, 0 };
    
    std::vector<SmallPos> attackable;

    std::vector<sf::Sprite> boardSprites;
    std::vector<sf::Sprite> boardPieces;
    std::vector<SmallPos> validMoves;

    bool rotate_board = false;
    float rotation = 180;
    bool start_rotation = false;

    sf::Clock deltaClock;
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::R) {
                    start_rotation = true;
                }
            }
            else if (event.type == sf::Event::MouseMoved)
            {
                prevBoardPos = boardPos;
                mousePos = sf::Vector2f(event.mouseMove.x - float(size.x) / 2, event.mouseMove.y - float(size.y) / 2);
                if (!rotate_board)
                    boardPos = SmallPos{ int8_t(BOARD_WIDTH / 2 - floor((mousePos.x) / (GRID_SIZE * SCALE)) - 1), int8_t(BOARD_HEIGHT / 2 - floor((mousePos.y) / (GRID_SIZE * SCALE)) - 1) };
                else
                    boardPos = SmallPos{ int8_t(floor((mousePos.x) / (GRID_SIZE* SCALE)) + BOARD_WIDTH / 2), int8_t(floor((mousePos.y) / (GRID_SIZE * SCALE)) + BOARD_HEIGHT / 2) };
                if (boardPos.x < 0) {
                    boardPos.x = 0;
                }
                if (boardPos.x >= N) {
                    boardPos.x = N - 1;
                }
                if (boardPos.y < 0) {
                    boardPos.y = 0;
                }
                if (boardPos.y >= N) {
                    boardPos.y = N - 1;
                }
                if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
                    simulation.AddDensity(boardPos.x, boardPos.y, 100);
                    simulation.AddVelocity(boardPos.x, boardPos.y, boardPos.x - prevBoardPos.x, boardPos.y - prevBoardPos.y);
                }
            }
            else if (event.type == sf::Event::MouseButtonPressed) {
                
            }
            else if (event.type == sf::Event::Resized)
            {
                size = window.getSize();
                centreView.setSize(sf::Vector2f(size.x, size.y));
                centreView.setCenter(0, 0);
            }
            else if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::Resized)
            {
                size = window.getSize();
                centreView.setSize(sf::Vector2f(size.x, size.y));
                centreView.setCenter(0, 0);
            }
        }
        if (start_rotation) {
            rotation += 2.f;
            if (rotation == 360)
                rotation = 0;
            if (rotation == 180 || rotation == 0) {
                rotate_board = !rotate_board;
                start_rotation = false;
            }
        }
        centreView.setRotation(rotation);

        window.setView(centreView);
        ImGui::SFML::Update(window, deltaClock.restart());
        window.clear();

        simulation.step();

        sprites.clear();
        drawState(simulation);
        for (auto& sprite : sprites) {
            window.draw(sprite);
        }
        ImGui::Begin("DebugWindow");
        ImGui::Text(std::string("MousePos: " + std::to_string(mousePos.x) + "x, "+std::to_string(mousePos.y) + "y").c_str());
        ImGui::Text(std::string("BoardPos: " + std::to_string(boardPos.x) + "x, "+std::to_string(boardPos.y) + "y").c_str());
        ImGui::End();
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    
    return 0;
}