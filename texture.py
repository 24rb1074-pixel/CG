import glfw
from pathlib import Path

from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

WINDOW_SIZE = 512


def initTextureFromFile(filename):
    """画像ファイルを読み込み、OpenGL のテクスチャを作成する。"""
    with Image.open(filename) as source:
        # glTexImage2D に渡す形式を RGB に統一する。
        image = source.convert("RGB")
        # PIL と OpenGL では画像の原点が上下逆なので反転する。
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        width, height = image.size
        data = image.tobytes()

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # テクスチャ画像はバイト単位に詰め込まれている。
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        width,
        height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        data,
    )

    # テクスチャの補間方法を指定する。
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id


def drawSquare():
    """現在選択されているテクスチャを貼った四角形を描画する。"""
    glEnable(GL_TEXTURE_2D)

    # 下地を白にして、テクスチャ本来の色で描画する。
    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-0.5, -0.5, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(0.5, -0.5, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(0.5, 0.5, 0.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-0.5, 0.5, 0.0)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def display(texture_id):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    gluLookAt(
        3.0, 4.0, 5.0,  # 視点の位置
        0.0, 0.0, 0.0,  # 注視点の位置
        0.0, 1.0, 0.0,  # 上方向ベクトル
    )

    glPushMatrix()
    glScalef(3.0, 3.0, 3.0)
    glRotatef(-20.0, 1.0, 0.0, 0.0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    drawSquare()
    glBindTexture(GL_TEXTURE_2D, 0)
    glPopMatrix()


def resize(window, width, height):
    del window
    width = max(width, 1)
    height = max(height, 1)
    glViewport(0, 0, width, height)

    # 3次元空間を画面へ投影する。
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def keyboard(window, key, scancode, action, mods):
    del scancode, mods
    if action == glfw.PRESS and key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
        glfw.set_window_should_close(window, True)


def init():
    glClearColor(0.2, 0.2, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)
    image_path = Path(__file__).with_name("mandrill.png")
    return initTextureFromFile(image_path)


def main():
    if not glfw.init():
        print("GLFW の初期化に失敗しました。")
        return

    window = glfw.create_window(
        WINDOW_SIZE, WINDOW_SIZE, "Texture Mapping", None, None
    )
    if window is None:
        print("ウィンドウの作成に失敗しました。")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, resize)
    glfw.set_key_callback(window, keyboard)

    texture_id = init()
    width, height = glfw.get_framebuffer_size(window)
    resize(window, width, height)

    try:
        while not glfw.window_should_close(window):
            display(texture_id)
            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        glDeleteTextures([texture_id])
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
