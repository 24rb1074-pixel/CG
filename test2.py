import glfw
from OpenGL.GL import *
import numpy as np

# 花の色 (初期値は緑)
current_color = [0.0, 0.6, 0.2]

# 移動、拡大縮小、回転の「現在値」
pos_x = 0.0
pos_y = 0.0
scale_val = 1.0
angle_deg = 0.0

# 1. 花の形を作る
def make_flower():

    num_petals = 6
    vertices = []

    base_petal = np.array([[0.0, 0.4], [-0.1, 0.05], [0.1, 0.05]], dtype=np.float32)
    
    for i in range(num_petals):
        # 360度を等分した角度
        theta = np.radians(i * (360 / num_petals))
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        for v in base_petal:
            nx = v[0] * cos_t - v[1] * sin_t
            ny = v[0] * sin_t + v[1] * cos_t
            vertices.append([nx, ny])
            
    return vertices

flower_vertices = make_flower()


# 2. 線形写像
def apply_transform(x, y):

    # 同次座標系 (x, y, 1.0) 
    v = np.array([x, y, 1.0])
    
    # S:拡大縮小行列
    S = np.array([[scale_val, 0.0, 0.0],
                  [0.0, scale_val, 0.0],
                  [0.0, 0.0, 1.0]])
    
    # R:回転行列
    rad = np.radians(angle_deg)
    R = np.array([[np.cos(rad), -np.sin(rad), 0.0],
                  [np.sin(rad),  np.cos(rad), 0.0],
                  [0.0,          0.0,         1.0]])
    
    # T:平行移動行列
    T = np.array([[1.0, 0.0, pos_x],
                  [0.0, 1.0, pos_y],
                  [0.0, 0.0, 1.0]])
    
    # M = T @ R @ S
    M = np.dot(T, np.dot(R, S))
    
    # 頂点に行列を適用
    v_new = np.dot(M, v)
    
    return v_new[0], v_new[1]


# 3. 描画処理 
def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(current_color[0], current_color[1], current_color[2])
    
    glBegin(GL_TRIANGLES)
    for vx, vy in flower_vertices:
        tx, ty = apply_transform(vx, vy)
        glVertex2f(tx, ty)
    glEnd()
    
    glFlush()


def init():
    glClearColor(1.0, 1.0, 1.0, 1.0)

def refresh(window):
    display()
    glfw.swap_buffers(window)


# 4. キーボード操作
def keyboard(window, key, scancode, action, mods):
    global current_color, pos_x, pos_y, scale_val, angle_deg
    
    if action == glfw.PRESS:
        if key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)
            print("Q key pressed - exiting")

        # 平行移動 (矢印キー)
        elif key == glfw.KEY_UP:
            pos_y += 0.05
            print("Up key pressed")
        elif key == glfw.KEY_DOWN:
            pos_y -= 0.05
            print("Down key pressed")
        elif key == glfw.KEY_RIGHT:
            pos_x += 0.05
            print("Right key pressed")
        elif key == glfw.KEY_LEFT:
            pos_x -= 0.05
            print("Left key pressed")

        # 拡大縮小 (Z と X キー)
        elif key == glfw.KEY_Z:
            scale_val += 0.1
            print("Z key pressed")
        elif key == glfw.KEY_X:
            scale_val -= 0.1
            print("X key pressed")

        # 回転 (A と D キー)
        elif key == glfw.KEY_A:
            angle_deg += 5.0
            print("A key pressed")
        elif key == glfw.KEY_D:
            angle_deg -= 5.0
            print("D key pressed")

        # 色変更 
        elif key == glfw.KEY_R:
            current_color = [1.0, 0.0, 0.0]
            print("R key pressed")
        elif key == glfw.KEY_G:
            current_color = [0.0, 1.0, 0.0]
            print("G key pressed")
        elif key == glfw.KEY_B:
            current_color = [0.0, 0.0, 1.0]
            print("B key pressed")

    refresh(window)

# 変更なし
def main():
    glfw.init()
    win = glfw.create_window(500, 500, "Interactive Flower", None, None)
    glfw.make_context_current(win)
    init()
    
    glfw.set_window_refresh_callback(win, refresh)
    glfw.set_key_callback(win, keyboard)
    
    refresh(win)
    
    while not glfw.window_should_close(win):
        glfw.wait_events()
        
    glfw.terminate()

if __name__ == "__main__":
    main()