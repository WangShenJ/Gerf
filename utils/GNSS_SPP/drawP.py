import pandas as pd
import open3d as o3d

if __name__ == '__main__':
    datac = pd.read_csv("../../data/phase-exp-1/dataTrain.csv")
    fP = open("../../data/pointCloud/position.txt", "w+", encoding='utf-8')
    fS = open("../../data/pointCloud/sPosition.txt", "w+", encoding='utf-8')
    datas = datac.values
    for data in datas:
        sx, sy, sz = data[3], data[4], data[5]
        x, y, z = data[0], data[1], data[2]
        s = str(sx) + " " + str(sy) + " " + str(sz) + "\n"
        fS.write(s)
        s = str(x) + " " + str(y) + " " + str(z) + "\n"
        fP.write(s)

    positionC = o3d.io.read_point_cloud("../../data/pointCloud/position.txt", format='xyz')
    positionS = o3d.io.read_point_cloud("../../data/pointCloud/sPosition.txt", format='xyz')
    positionC.paint_uniform_color([1, 0, 0])
    positionS.paint_uniform_color([0, 0, 1])
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=6378137.0,
                                                          resolution=100)
    o3d.visualization.draw_geometries([positionC,positionS], width=600, height=600)
