import trimesh
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

def drawAxis():
    lighting_enabled = glIsEnabled(GL_LIGHTING) # ライティングが有効かどうかをチェック

    glDisable(GL_LIGHTING) # 軸は光の影響を受けないようにする
    # xyz axis
    glBegin(GL_LINES)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(2, 0, 0) # 少し長くしました

    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 2, 0)

    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 2)
    glEnd()

    if lighting_enabled:
        glEnable(GL_LIGHTING) # 軸を描き終わったらライティングをオンに戻す
    else:
        glDisable(GL_LIGHTING) # ライティングが元々オフだった場合はオフのままにする

def drawTeapot():
    # teapot.obj を読み込む
    mesh = trimesh.load("teapot.obj")
    # 頂点の総数
    num_vertices = len(mesh.vertices)
    print(f"Number of vertices: {num_vertices}")
    # 面の総数
    num_faces = len(mesh.faces)
    print(f"Number of faces: {num_faces}")
    glBegin(GL_TRIANGLES)
    for i in range(num_faces):
        for j in range(3):  # 三角形の各頂点に対して
            # i 番目の面の j 番目の頂点の位置を取得
            vertex_index = mesh.faces[i][j]
            vertex_position = mesh.vertices[vertex_index]
            # i 番目の面の j 番目の頂点の法線を取得
            vertex_normal = mesh.vertex_normals[vertex_index]
            # 法線を設定
            glNormal3fv(vertex_normal)
            # 頂点を設定
            glVertex3fv(vertex_position)
    glEnd()
    '''i, j = 0, 0
    # 通し番号で i 番目の頂点の位置
    p = mesh.vertices[i]
    # i 番目の面の頂点の数
    num_vertices_in_face = len(mesh.faces[i])
    # i 番目の面の j 番目の頂点の位置
    p = mesh.vertices[mesh.faces[i][j]]
    # 通し番号で i 番目の頂点の法線
    n = mesh.vertex_normals[i]
    # i 番目三角形の j 番目の頂点の法線
    n = mesh.vertex_normals[mesh.faces[i][j]]'''

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # 視点の設定
    gluLookAt(3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    # 光源位置を(3, 0, 0)に指定
    lpos = np.array([3.0, 0.0, 0.0, 1.0], dtype=np.float32)
    glLightfv(GL_LIGHT0, GL_POSITION, lpos)
    
    drawAxis()
    drawTeapot()
    drawPlane(10)  # 10x10の平面を描画
    

def perspective(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(30.0, width / height, 1.0, 100.0)
    glMatrixMode(GL_MODELVIEW)

def init():
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.5, 0.5, 0.5, 1.0) # 陰影が見やすいように背景を少し暗くしました
    perspective(512, 512)

    # シェーディング（ライティング）の有効化
    glEnable(GL_LIGHTING)
    # 0番目のライトを有効化
    glEnable(GL_LIGHT0)

def main():
    if not glfw.init():
        return
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(512, 512, "Lighting Test", None, None)
    if not window:
        glfw.terminate()
        return
        
    glfw.make_context_current(window)
    init()
    
    glfw.set_framebuffer_size_callback(window, lambda w, width, height: perspective(width, height))
    
    while not glfw.window_should_close(window):
        display()
        glfw.swap_buffers(window)
        glfw.poll_events()
        
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()