
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "bb_spatial.h"
#include "audio_engine.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

static const float WORLD   = 100.0f;
static const int   PANEL_L = 300;       
static const int   PANEL_R = 300;       
static const float PI      = 3.14159265f;

struct RoomDef {
    const char* name;
    float ox,oy,oz, ex,ey,ez;
    uint16_t ir;
    float wet_level;  
    float abs[6];
    float cr,cg,cb;   
};

static const RoomDef ROOMS[2] = {
    { "Cathedral", 5.f,0.f,5.f, 20.f,8.f,15.f, 1, 0.85f, {0.03f,0.03f,0.02f,0.02f,0.02f,0.02f}, 0.22f, 0.48f, 0.95f },
    { "Corridor", 11.f,0.f,20.f, 3.f,4.f,10.f, 2, 0.25f, {0.40f,0.40f,0.55f,0.55f,0.20f,0.20f}, 0.95f, 0.40f, 0.15f },
};

static const float POX=11.f,POY=0.f,POZ=19.f, PEX=3.f, PEY=4.f, PEZ=1.f;

static bb::LinearOctree buildScene() {
    bb::LinearOctree tree(10);
    for (auto& r : ROOMS) {
        bb::AcousticParams p;
        p.ir_primary=r.ir; p.ir_secondary=0xFFFF; p.blend_t=0.f; p.wet_level=r.wet_level;
        for(int i=0; i<6; ++i) p.absorption[i] = bb::quantize_abs(r.abs[i]);
        bb::fill_box(tree,r.ox,r.oy,r.oz,r.ex,r.ey,r.ez,p,WORLD);
    }
    bb::fill_portal(tree,POX,POY,POZ,PEX,PEY,PEZ, 1,2,2, ROOMS[0].wet_level, ROOMS[1].wet_level, ROOMS[0].abs,ROOMS[1].abs,WORLD);
    tree.bake(); return tree;
}

struct SoundSource { float x,y,z,r,g,b; uint16_t ir_id; const char* name; };

static void mmul(const float* a, const float* b, float* out) {
    for (int col=0;col<4;col++) for (int row=0;row<4;row++) {
        float s=0; for (int k=0;k<4;k++) s+=a[k*4+row]*b[col*4+k];
        out[col*4+row]=s;
    }
}

static void lookAt(float ex,float ey,float ez, float tx,float ty,float tz, float* m) {
    float fx=tx-ex,fy=ty-ey,fz=tz-ez; float fl=std::sqrt(fx*fx+fy*fy+fz*fz); fx/=fl;fy/=fl;fz/=fl;
    float rx=fz,ry=0.f,rz=-fx; float rl=std::sqrt(rx*rx+ry*ry+rz*rz); if(rl>1e-6f){rx/=rl;ry/=rl;rz/=rl;}
    float ux=ry*fz-rz*fy, uy=rz*fx-rx*fz, uz=rx*fy-ry*fx;
    m[ 0]=rx; m[ 4]=ry; m[ 8]=rz; m[12]=-(rx*ex+ry*ey+rz*ez); m[ 1]=ux; m[ 5]=uy; m[ 9]=uz; m[13]=-(ux*ex+uy*ey+uz*ez);
    m[ 2]=-fx; m[ 6]=-fy; m[10]=-fz; m[14]= (fx*ex+fy*ey+fz*ez); m[ 3]=0; m[ 7]=0; m[11]=0; m[15]=1;
}

static void proj(float* m, float fovDeg, float aspect, float zn, float zf) {
    float f=1.f/std::tan(fovDeg*PI/360.f); memset(m,0,64);
    m[0]=f/aspect; m[5]=f; m[10]=(zf+zn)/(zn-zf); m[11]=-1.f; m[14]=2.f*zf*zn/(zn-zf);
}

struct Camera {
    float pitch=-25.f, yaw=40.f, dist=45.f; float tx=15.f, ty=4.f, tz=14.f;
    void matrix(float* m) const {
        float p=pitch*PI/180.f, y=yaw*PI/180.f;
        lookAt(tx+dist*std::cos(p)*std::sin(y), ty+dist*std::sin(p), tz+dist*std::cos(p)*std::cos(y), tx,ty,tz, m);
    }
};

static const char* LINE_VS = R"(#version 130
in vec3 aPos; in vec3 aCol; uniform mat4 uMVP; out vec3 vCol;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); vCol = aCol; })";
static const char* LINE_FS = R"(#version 130
in vec3 vCol; out vec4 fragColor;
void main() { fragColor = vec4(vCol, 1.0); })";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s=glCreateShader(type); glShaderSource(s,1,&src,nullptr); glCompileShader(s); return s;
}

struct LineRenderer {
    GLuint prog=0, vao=0, vbo=0; std::vector<float> buf;
    void init() {
        GLuint vs=compileShader(GL_VERTEX_SHADER, LINE_VS), fs=compileShader(GL_FRAGMENT_SHADER, LINE_FS);
        prog=glCreateProgram(); glAttachShader(prog,vs); glAttachShader(prog,fs); glLinkProgram(prog); glDeleteShader(vs); glDeleteShader(fs);
        glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo); glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,24,(void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,24,(void*)12); glBindVertexArray(0);
    }
    void add(float x0,float y0,float z0, float x1,float y1,float z1, float r, float g, float b) { buf.insert(buf.end(), {x0,y0,z0,r,g,b, x1,y1,z1,r,g,b}); }
    void addBox(float ox,float oy,float oz, float ex,float ey,float ez, float r, float g, float b) {
        float x1=ox+ex, y1=oy+ey, z1=oz+ez;
        add(ox,oy,oz, x1,oy,oz, r,g,b); add(x1,oy,oz, x1,oy,z1, r,g,b); add(x1,oy,z1, ox,oy,z1, r,g,b); add(ox,oy,z1, ox,oy,oz, r,g,b);
        add(ox,y1,oz, x1,y1,oz, r,g,b); add(x1,y1,oz, x1,y1,z1, r,g,b); add(x1,y1,z1, ox,y1,z1, r,g,b); add(ox,y1,z1, ox,y1,oz, r,g,b);
        add(ox,oy,oz, ox,y1,oz, r,g,b); add(x1,oy,oz, x1,y1,oz, r,g,b); add(x1,oy,z1, x1,y1,z1, r,g,b); add(ox,oy,z1, ox,y1,z1, r,g,b);
    }
    void flush(const float* mvp, float lineWidth=1.5f) {
        if (buf.empty()) return; glBindBuffer(GL_ARRAY_BUFFER,vbo); glBufferData(GL_ARRAY_BUFFER, buf.size()*4, buf.data(), GL_DYNAMIC_DRAW);
        glUseProgram(prog); glUniformMatrix4fv(glGetUniformLocation(prog,"uMVP"), 1,GL_FALSE,mvp);
        glBindVertexArray(vao); glLineWidth(lineWidth); glDrawArrays(GL_LINES,0,(GLsizei)(buf.size()/6)); glBindVertexArray(0); buf.clear();
    }
};

static std::vector<float> makeIR(float rt60, int n=200) {
    std::vector<float> out(n); std::mt19937 rng(42); std::uniform_real_distribution<float> d(-1.f,1.f);
    for (int i=0;i<n;i++) out[i] = d(rng)*std::exp(-6.91f*(float)i/(float)n/std::max(0.01f,rt60));
    return out;
}

struct Perf {
    double qNs=0, mNs=0; int hits=0, miss=0, total=0;
    void update(bool same, double q, double mn) { qNs =qNs *0.95+q *0.05; mNs =mNs *0.95+mn*0.05; total++; same?hits++:miss++; }
    float hitPct() const { return total?100.f*hits/total:0.f; }
};

struct App {
    bb::LinearOctree tree; Camera cam; LineRenderer lr; Perf perf; AudioEngine audio;
    float lx=15.f, ly=2.f, lz=12.f, moveSpeed=5.f; uint32_t prevM=0; int divLevels=0; bool inRoom=false;
    const bb::Leaf* leaf=nullptr; bool orbiting=false; double omx=0, omy=0;
    struct TP { float x,y,z; }; std::vector<TP> trail;
    std::vector<SoundSource> sources = { {15.f, 4.f, 12.5f, 0.2f, 0.5f, 1.0f, 1, "Cathedral Center"}, {12.5f, 2.f, 25.f, 1.0f, 0.4f, 0.2f, 2, "Corridor End"} };
    App() : tree(buildScene()) {}
};

static App* G = nullptr;

static void applyStyle() {
    ImGuiStyle& s = ImGui::GetStyle(); s.WindowRounding = 0.f; s.FrameRounding = 4.f; s.WindowBorderSize = 0.f; s.ItemSpacing = {8,5};
    auto* c = ImGui::GetStyle().Colors; c[ImGuiCol_WindowBg] = {0.09f,0.09f,0.11f,1.f}; c[ImGuiCol_TitleBg] = {0.07f,0.07f,0.09f,1.f};
    c[ImGuiCol_FrameBg] = {0.14f,0.14f,0.18f,1.f}; c[ImGuiCol_Button] = {0.18f,0.32f,0.62f,0.8f}; c[ImGuiCol_Text] = {0.88f,0.88f,0.90f,1.f};
}

static void colorBar(float value, ImVec4 col, float h=8.f) {
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col); ImGui::ProgressBar(value, ImVec2(-1.f, h), ""); ImGui::PopStyleColor();
}

static void drawPanelListener(App& a, int winH) {
    ImGui::SetNextWindowPos(ImVec2(0.f, 0.f), ImGuiCond_Always); 
    ImGui::SetNextWindowSize(ImVec2((float)PANEL_L, (float)winH), ImGuiCond_Always);
    ImGui::Begin("##left", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::TextDisabled("LISTENER"); ImGui::Separator();
    ImGui::Text("Pos   %.1f, %.1f, %.1f m", a.lx,a.ly,a.lz);
    uint32_t m = bb::morton_encode(bb::world_to_grid(a.lx,WORLD,1024), bb::world_to_grid(a.ly,WORLD,1024), bb::world_to_grid(a.lz,WORLD,1024));
    ImGui::Spacing(); ImGui::TextDisabled("MORTON CODE"); ImGui::TextColored(ImVec4(1.f,0.80f,0.18f,1.f), "0x%08X", m);
    
    ImGui::Spacing(); ImGui::TextDisabled("bit layout  [x y z  x y z ...]");
    ImDrawList* dl = ImGui::GetWindowDrawList(); ImVec2 cp = ImGui::GetCursorScreenPos();
    const float BW=7.4f, BH=12.f, GAP=1.1f;
    static const ImVec4 ON[3] = {{0.28f,0.52f,1.f,1},{0.30f,0.80f,0.36f,1},{1.f,0.48f,0.16f,1}}, OFF = {0.16f,0.16f,0.20f,1};
    for(int i=29;i>=0;i--){ ImVec4 col=((m>>i)&1)?ON[i%3]:OFF; float bx=cp.x+(29-i)*(BW+GAP); dl->AddRectFilled(ImVec2(bx,cp.y),ImVec2(bx+BW,cp.y+BH), ImGui::ColorConvertFloat4ToU32(col),2.f); }
    ImGui::Dummy(ImVec2(30.f*(BW+GAP), BH+4.f)); ImGui::TextDisabled("  blue=X  green=Y  orange=Z");
    ImGui::Spacing();

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    ImGui::TextDisabled("OCTREE STATS"); ImGui::Separator();
    ImGui::Text("Leaves     %zu  (%zu KB)", a.tree.size(), a.tree.size()*sizeof(bb::Leaf)/1024);
    ImGui::Text("Lookup     %.1f ns", a.perf.qNs); colorBar(a.perf.hitPct()/100.f, ImVec4(0.26f,0.72f,0.42f,1.f), 7.f);
    
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    ImGui::TextDisabled("ACOUSTIC ZONE"); ImGui::Separator();
    if (!a.inRoom || !a.leaf) { ImGui::TextDisabled("Outside all zones"); } 
    else {
        float bl = a.leaf->params.blend_t;
        ImGui::TextColored(bl<0.05f ? ImVec4{0.28f,0.52f,1.f,1.f} : bl>0.95f ? ImVec4{0.95f,0.40f,0.14f,1.f} : ImVec4{0.82f,0.72f,0.20f,1.f}, 
                           "%s", bl<0.05f?"Cathedral":bl>0.95f?"Corridor":"Portal Blending");
        colorBar(bl, ImVec4(0.26f+bl*0.64f, 0.52f-bl*0.12f, 0.96f-bl*0.82f, 1.f), 9.f); ImGui::Text("Blend: %.3f", bl);
        
        ImGui::Spacing();
        ImGui::TextDisabled("ABSORPTION COEFFICIENTS");
        static const char* FL[6]={"-X","+X","-Y","+Y","-Z","+Z"};
        static const ImVec4 FC[6]={ {0.82f,0.26f,0.26f,1},{0.82f,0.26f,0.26f,1}, {0.26f,0.78f,0.26f,1},{0.26f,0.78f,0.26f,1}, {0.26f,0.46f,0.90f,1},{0.26f,0.46f,0.90f,1} };
        for (int i=0;i<6;i++) { 
            float abs_val = bb::dequantize_abs(a.leaf->params.absorption[i]);
            ImGui::Text("%-3s  %.3f", FL[i], abs_val); ImGui::SameLine(105.f); colorBar(abs_val, FC[i], 7.f); 
        }
    }
    ImGui::End();
}

static void drawPanelDSP(App& a, int fbW, int fbH) {
    ImGui::SetNextWindowPos(ImVec2((float)(fbW - PANEL_R), 0.f), ImGuiCond_Always); 
    ImGui::SetNextWindowSize(ImVec2((float)PANEL_R, (float)fbH), ImGuiCond_Always);
    ImGui::Begin("##right", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::TextDisabled("DSP ENGINE"); ImGui::Separator();
    
    if (a.leaf) {
        ImGui::Text("Wet Level: %.2f", a.leaf->params.wet_level);
        float rt60 = 4.2f + (0.35f-4.2f) * a.leaf->params.blend_t; 
        
        std::vector<float> ir = makeIR(rt60/4.2f, 200);
        ImGui::PlotLines("##ir",ir.data(),(int)ir.size(), 0,nullptr,-1.f,1.f,ImVec2(-1.f,70.f)); 
        ImGui::Text("RT60 Decay  %.2f s", rt60);
    }
    
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    
    int ui_conv_len = a.audio.state.conv_length.load();
    ImGui::Text("FIR Block Length:");
    if (ImGui::SliderInt("##convlen", &ui_conv_len, 512, 16384, "%d samples")) {
        a.audio.state.conv_length.store(ui_conv_len);
    }
    ImGui::TextDisabled("Actual render time: %.1f ms", (float)ui_conv_len / 48.0f);
    
    if (ui_conv_len > 8192) ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "WARNING: High CPU Load!");
    ImGui::End();
}

static void draw3D(App& a, int fbW, int fbH) {
    int vpX = PANEL_L, vpW = fbW - PANEL_L - PANEL_R; if (vpW <= 0) return;
    glViewport(vpX, 0, vpW, fbH); glEnable(GL_SCISSOR_TEST); glScissor(vpX, 0, vpW, fbH);
    glClearColor(0.06f,0.06f,0.08f,1.f); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); glDisable(GL_SCISSOR_TEST);

    float P[16], V[16], MVP[16]; proj(P, 55.f, (float)vpW/(float)fbH, 0.3f, 500.f); a.cam.matrix(V); mmul(P, V, MVP);

    for (int i=0;i<=40;i+=2) { float f=(float)i; a.lr.add(f,0,0, f,0,40, 0.12f,0.12f,0.15f); a.lr.add(0,0,f, 40,0,f, 0.12f,0.12f,0.15f); }
    for (auto& r : ROOMS) {
        bool active = a.inRoom && a.leaf && a.leaf->params.ir_primary == r.ir;
        float k = active ? 1.f : 0.25f;
        a.lr.addBox(r.ox,r.oy,r.oz, r.ex,r.ey,r.ez, r.cr*k, r.cg*k, r.cb*k);
        float cx=r.ox+r.ex*.5f, cz=r.oz+r.ez*.5f, lk=k*0.5f;
        a.lr.add(cx-1.2f,0.01f,cz, cx+1.2f,0.01f,cz, r.cr*lk,r.cg*lk,r.cb*lk); a.lr.add(cx,0.01f,cz-1.2f, cx,0.01f,cz+1.2f, r.cr*lk,r.cg*lk,r.cb*lk);
    }
    
    float bl = (a.inRoom&&a.leaf) ? a.leaf->params.blend_t : 0.f;
    a.lr.addBox(POX,POY,POZ, PEX,PEY,PEZ, 0.82f-bl*0.54f, 0.76f-bl*0.36f, 0.16f+bl*0.74f);

    
    
    if (a.leaf) {
        float vs = WORLD / 1024.0f; 
        const bb::Leaf* base = a.tree.leaves().data();
        intptr_t center_idx = a.leaf - base;
        
        for (intptr_t i = -4; i <= 4; ++i) {
            intptr_t idx = center_idx + i;
            if (idx >= 0 && idx < (intptr_t)a.tree.size()) {
                const bb::Leaf& neighbor = base[idx];
                uint32_t gx, gy, gz; bb::morton_decode(neighbor.morton, gx, gy, gz);
                float wx = gx * vs, wy = gy * vs, wz = gz * vs;
                if (i == 0) {
                    a.lr.addBox(wx, wy, wz, vs, vs, vs, 0.2f, 0.9f, 0.9f); 
                } else {
                    float intensity = 0.8f - std::abs((float)i) * 0.15f;
                    a.lr.addBox(wx, wy, wz, vs, vs, vs, 0.2f*intensity, 0.5f*intensity, 0.9f*intensity); 
                }
            }
        }
    }

    for (size_t i=1;i<a.trail.size();i++) { float fade=(float)i/a.trail.size(); a.lr.add(a.trail[i-1].x,a.trail[i-1].y,a.trail[i-1].z, a.trail[i].x,a.trail[i].y,a.trail[i].z, 0.40f*fade, 0.72f*fade, 0.40f*fade); }
    
    for (const auto& s : a.sources) {
        float size = 0.4f;
        a.lr.add(s.x-size,s.y,s.z, s.x+size,s.y,s.z, s.r,s.g,s.b); a.lr.add(s.x,s.y-size,s.z, s.x,s.y+size,s.z, s.r,s.g,s.b);
        a.lr.add(s.x,s.y,s.z-size, s.x,s.y,s.z+size, s.r,s.g,s.b); a.lr.add(s.x,s.y,s.z, s.x,0.0f,s.z, s.r*0.3f,s.g*0.3f,s.b*0.3f);
    }

    float lx=a.lx, ly=a.ly, lz=a.lz, cr=a.inRoom?0.96f:0.40f, cg=a.inRoom?0.96f:0.40f, cb=a.inRoom?0.22f:0.40f, D=0.55f;
    a.lr.add(lx-D,ly,lz, lx+D,ly,lz, cr,cg,cb); a.lr.add(lx,ly-D,lz, lx,ly+D,lz, cr,cg,cb); a.lr.add(lx,ly,lz-D, lx,ly,lz+D, cr,cg,cb);
    const int RS=32; float R=0.9f;
    for (int i=0;i<RS;i++) {
        float a0=2.f*PI*i/RS, a1=2.f*PI*(i+1)/RS;
        a.lr.add(lx+std::cos(a0)*R, ly, lz+std::sin(a0)*R, lx+std::cos(a1)*R, ly, lz+std::sin(a1)*R, cr*0.65f, cg*0.65f, cb*0.65f);
    }
    a.lr.flush(MVP, 1.5f);
}

int main() {
    glfwInit(); glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow* win = glfwCreateWindow(1440, 900, "bb_spatial  —  Spatial Audio Observability", NULL, NULL);
    glfwMakeContextCurrent(win); glfwSwapInterval(1); gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    ImGui::CreateContext(); applyStyle(); ImGui_ImplGlfw_InitForOpenGL(win, false); ImGui_ImplOpenGL3_Init("#version 130");

    App app; G = &app; app.lr.init(); glEnable(GL_DEPTH_TEST);
    app.audio.init("audio/source/flute.wav", "audio/ir/cathedral.wav", "audio/ir/corridor.wav");

    glfwSetMouseButtonCallback(win, [](GLFWwindow* w, int b, int a, int mods){
        ImGui_ImplGlfw_MouseButtonCallback(w, b, a, mods);   
        if (ImGui::GetIO().WantCaptureMouse) return;          
        if (b==GLFW_MOUSE_BUTTON_MIDDLE) { G->orbiting=(a==GLFW_PRESS); glfwGetCursorPos(w, &G->omx, &G->omy); }
    });
    glfwSetCursorPosCallback(win, [](GLFWwindow* w, double x, double y){
        ImGui_ImplGlfw_CursorPosCallback(w, x, y);
        if (!G->orbiting) return;
        G->cam.yaw  += (float)(x-G->omx)*0.4f;
        G->cam.pitch = std::clamp((float)(G->cam.pitch+(y-G->omy)*0.4f),-89.f,89.f);
        G->omx=x; G->omy=y;
    });
    glfwSetKeyCallback(win, [](GLFWwindow* w, int k, int s, int a, int m){ ImGui_ImplGlfw_KeyCallback(w,k,s,a,m); });
    glfwSetCharCallback(win, [](GLFWwindow* w, unsigned int c){ ImGui_ImplGlfw_CharCallback(w,c); });
    glfwSetScrollCallback(win, [](GLFWwindow* w, double dx, double dy){
        ImGui_ImplGlfw_ScrollCallback(w, dx, dy);
        if (!ImGui::GetIO().WantCaptureMouse) G->cam.dist = std::clamp(G->cam.dist-(float)dy*2.f, 4.f, 180.f);
    });
    
    double lastT = glfwGetTime();
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents(); double now = glfwGetTime(); float dt = (float)(now-lastT); lastT = now;

        if (!ImGui::GetIO().WantCaptureKeyboard) {
            float sp = app.moveSpeed * dt, yr = app.cam.yaw * PI / 180.f;
            float fx = std::sin(yr), fz = std::cos(yr), rx = std::cos(yr), rz = -std::sin(yr);
            if(glfwGetKey(win,GLFW_KEY_W)==1) { app.lx += fx*sp; app.lz += fz*sp; }
            if(glfwGetKey(win,GLFW_KEY_S)==1) { app.lx -= fx*sp; app.lz -= fz*sp; }
            if(glfwGetKey(win,GLFW_KEY_A)==1) { app.lx -= rx*sp; app.lz -= rz*sp; }
            if(glfwGetKey(win,GLFW_KEY_D)==1) { app.lx += rx*sp; app.lz += rz*sp; }
            if(glfwGetKey(win,GLFW_KEY_Q)==1) { app.ly += sp; }
            if(glfwGetKey(win,GLFW_KEY_E)==1) { app.ly -= sp; }
            app.lx = std::clamp(app.lx, 0.1f, 99.9f); app.ly = std::clamp(app.ly, 0.1f, 99.9f); app.lz = std::clamp(app.lz, 0.1f, 99.9f);
        }

        app.cam.tx=app.lx; app.cam.ty=app.ly; app.cam.tz=app.lz;

        uint32_t m = bb::morton_encode(bb::world_to_grid(app.lx,WORLD,1024), bb::world_to_grid(app.ly,WORLD,1024), bb::world_to_grid(app.lz,WORLD,1024));
        auto t0=std::chrono::high_resolution_clock::now(); app.leaf = app.tree.find(m);
        auto t1=std::chrono::high_resolution_clock::now();
        
        bool same = (m == app.prevM);
        app.perf.update(same, std::chrono::duration<double,std::nano>(t1-t0).count(), 0);
        app.divLevels = same ? 0 : bb::divergence_levels(app.prevM, m, 10);
        app.prevM = m; app.inRoom = (app.leaf!=nullptr);

        if (!same) { app.trail.push_back(App::TP{app.lx,app.ly,app.lz}); if (app.trail.size()>200) app.trail.erase(app.trail.begin()); }

        app.audio.state.lx.store(app.lx, std::memory_order_relaxed);
        app.audio.state.ly.store(app.ly, std::memory_order_relaxed);
        app.audio.state.lz.store(app.lz, std::memory_order_relaxed);
        if (app.leaf) {
            app.audio.state.active_ir.store(app.leaf->params.ir_primary, std::memory_order_relaxed);
            app.audio.state.blend_t.store(app.leaf->params.blend_t, std::memory_order_relaxed);
            app.audio.state.wet_level.store(app.leaf->params.wet_level, std::memory_order_relaxed);
        }

        int fbW,fbH; glfwGetFramebufferSize(win,&fbW,&fbH); draw3D(app, fbW, fbH);
        glViewport(0,0,fbW,fbH); ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
        drawPanelListener(app, fbH); drawPanelDSP(app, fbW, fbH);
        ImGui::Render(); ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); glfwSwapBuffers(win);
    }
    return 0;
}