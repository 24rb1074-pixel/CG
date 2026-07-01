import glfw
from OpenGL.GL import * # type: ignore
from OpenGL.GLU import * # type: ignore

vertex = [
    (0.0, 0.0, 0.0),  # A
    (1.0, 0.0, 0.0),  # B
    (1.0, 1.0, 0.0),  # C
    (0.0, 1.0, 0.0),  # D
    (0.0, 0.0, 1.0),  # E
    (1.0, 0.0, 1.0),  # F
    (1.0, 1.0, 1.0),  # G
    (0.0, 1.0, 1.0),  # H
]

edge = [
    (0, 1),  # ア (A-B)
    (1, 2),  # イ (B-C)
    (2, 3),  # ウ (C-D)
    (3, 0),  # エ (D-A)
    (4, 5),  # オ (E-F)
    (5, 6),  # カ (F-G)
    (6, 7),  # キ (G-H)
    (7, 4),  # ク (H-E)
    (0, 4),  # ケ (A-E)
    (1, 5),  # コ (B-F)
    (2, 6),  # サ (C-G)
    (3, 7),  # シ (D-H)
]

# グローバル変数
pos = [0.0, 0.0, 0.0]
scale = 1.0
angle = 0.0
color = [0.0, 0.0, 0.0]


def display():
    glClear(GL_COLOR_BUFFER_BIT)

    # モデルビュー変換行列の設定
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # カメラの位置と向きを設定
    gluLookAt(3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    # モデル変換
    glTranslatef(pos[0], pos[1], pos[2])
    glRotatef(angle, 0.0, 1.0, 0.0)
    glScalef(scale, scale, scale)

    # 図形の描画
    glColor3fv(color)
    glBegin(GL_LINES)
    for i in range(12):
        glVertex3fv(vertex[edge[i][0]])
        glVertex3fv(vertex[edge[i][1]])
    glEnd()

# ウィンドウのリサイズと透視変換の設定
def perspective(width, height):
    glViewport(0, 0, width, height)
    
    # 透視変換行列の設定
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(30.0, width / height, 1.0, 100.0)
    
    # モデルビュー変換行列の設定
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    perspective(512, 512)
    
# フレームバッファサイズ変更時のコールバックを登録
def refresh(window):
    display()
    glfw.swap_buffers(window)

# キーボード入力のコールバックを登録
def keyboard(window, key, scancode, action, mods):
    global pos, scale, angle, color
    
    if action == glfw.PRESS:
        # 終了
        if key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)
            print("Q key pressed - exiting")

        # 平行移動 (矢印キー)
        elif key == glfw.KEY_UP:
            pos[1] += 0.05
            print("Up key pressed")
        elif key == glfw.KEY_DOWN:
            pos[1] -= 0.05
            print("Down key pressed")
        elif key == glfw.KEY_RIGHT:
            pos[0] += 0.05
            print("Right key pressed")
        elif key == glfw.KEY_LEFT:
            pos[0] -= 0.05
            print("Left key pressed")

        # 拡大縮小 (Z と X キー)
        elif key == glfw.KEY_Z:
            scale *= 1.1
            print("Z key pressed")
        elif key == glfw.KEY_X:
            scale *= 0.9
            print("X key pressed")

        # 回転 (A と D キー)
        elif key == glfw.KEY_A:
            angle += 5.0
            print("A key pressed")
        elif key == glfw.KEY_D:
            angle -= 5.0
            print("D key pressed")

        # 色変更 
        elif key == glfw.KEY_R:
            color = [1.0, 0.0, 0.0]
            print("R key pressed")
        elif key == glfw.KEY_G:
            color = [0.0, 1.0, 0.0]
            print("G key pressed")
        elif key == glfw.KEY_B:
            color = [0.0, 0.0, 1.0]
            print("B key pressed")

    refresh(window)

# ウィンドウのリサイズと透視変換の設定
def resize(window, width, height):
    perspective(width, height)

# 前回のコードを流用
def main():
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    window = glfw.create_window(512, 512, "Hello", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    init()

    # フレームバッファサイズ変更時のコールバックを登録
    glfw.set_framebuffer_size_callback(window, resize)

    # 起動時に一度呼んで投影行列を初期化
    resize(window, *glfw.get_framebuffer_size(window))
    
    # キーボード入力のコールバックを登録
    glfw.set_window_refresh_callback(window, refresh)
    glfw.set_key_callback(window, keyboard)

    while not glfw.window_should_close(window):
        display()
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
