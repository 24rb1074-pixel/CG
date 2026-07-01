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

def from_spherical(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def drawSphere(radius):
    # 【修正2】分割数を定義
    tdiv = 16 # 滑らかにするために少し分割数を増やしました
    pdiv = 32
    
    # 材質（マテリアル）の設定
    matRed = np.array([1, 0, 0, 1], dtype=np.float32)
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, matRed)

    matWhite = np.array([1, 1, 1, 1], dtype=np.float32)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, matWhite)
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0)

    glBegin(GL_QUADS)
    for i in range(tdiv):
        for j in range(pdiv):
            theta = i / tdiv * np.pi
            phi = j / pdiv * 2.0 * np.pi
            dtheta = 1.0 / tdiv * np.pi
            dphi = 1.0 / pdiv * 2.0 * np.pi
            
            v0 = from_spherical(radius, theta, phi)
            v1 = from_spherical(radius, theta + dtheta, phi)
            v2 = from_spherical(radius, theta + dtheta, phi + dphi)
            v3 = from_spherical(radius, theta, phi + dphi)
            
            # 球の法線は頂点位置の単位ベクトル（原点からの方向）
            # 【重要】ライティングにはこの法線（glNormal）が必須です！
            n0 = v0 / np.linalg.norm(v0)
            n1 = v1 / np.linalg.norm(v1)
            n2 = v2 / np.linalg.norm(v2)
            n3 = v3 / np.linalg.norm(v3)
            
            glNormal3fv(n0); glVertex3fv(v0)
            glNormal3fv(n1); glVertex3fv(v1)
            glNormal3fv(n2); glVertex3fv(v2)
            glNormal3fv(n3); glVertex3fv(v3)
    glEnd()

def drawPlaneY(size):
    # 【追加】床が球体のようにテカテカ光らないよう、光沢(SPECULAR)をゼロにリセットする
    matBlack = np.array([0, 0, 0, 1], dtype=np.float32)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, matBlack)
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)

    for i in range(int(size)):
        for j in range(int(size)):
            x = i - 0.5 * size
            z = j - 0.5 * size
            
            # 【変更】glColor3f の代わりに、マテリアル（材質）を設定する
            if (i + j) % 2 == 0:
                mat = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32) # 白
            else:
                mat = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32) # ダークグレー
            
            # 拡散反射光（基本の色）としてマテリアルを適用
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat)
            
            glBegin(GL_QUADS)
            # 【重要】床の面が「上(Y軸方向)」を向いているとOpenGLに教える
            glNormal3f(0.0, 1.0, 0.0) 
            
            glVertex3f(x, 0, z)
            glVertex3f(x + 1, 0, z)
            glVertex3f(x + 1, 0, z + 1)
            glVertex3f(x, 0, z + 1)
            glEnd()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # 視点の設定
    gluLookAt(3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    # 光源位置を(3, 0, 0)に指定
    lpos = np.array([3.0, 0.0, 0.0, 1.0], dtype=np.float32)
    glLightfv(GL_LIGHT0, GL_POSITION, lpos)
    
    glPushMatrix()
    glTranslatef(0.0, 1.0, 0.0)
    # ライティング中は glColor3f は無視されるため削除しました
    drawSphere(0.5)
    glPopMatrix()

    drawPlaneY(10)  # 10x10の平面を描画
    drawAxis()

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