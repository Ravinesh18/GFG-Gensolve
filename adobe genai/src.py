import os
os.environ['path'] += r';C:\Program Files\UniConvertor-2.0rc5\dlls'
import numpy as np
import matplotlib.pyplot as plt
import cv2
import svgwrite
import cairosvg
import csv

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

def detect_lines(image, paths_XYs):
    for XYs in paths_XYs:
        for XY in XYs:
            cv2.polylines(image, [XY.astype(np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
    return image

def detect_circles(image, paths_XYs):
    for XYs in paths_XYs:
        for XY in XYs:
            if len(XY) > 2:
                (x, y), radius = cv2.minEnclosingCircle(XY.astype(np.int32))
                if np.all(np.isclose(np.linalg.norm(XY - [x, y], axis=1), radius, rtol=0.1)):
                    cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    return image

def detect_rectangles(image, paths_XYs):
    for XYs in paths_XYs:
        for XY in XYs:
            if len(XY) == 4:
                cv2.polylines(image, [XY.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    return image

def detect_polygons(image, paths_XYs):
    for XYs in paths_XYs:
        for XY in XYs:
            if len(XY) > 4:
                cv2.polylines(image, [XY.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    return image

def detect_star_shapes(image, paths_XYs):
    for XYs in paths_XYs:
        for XY in XYs:
            hull = cv2.convexHull(XY.astype(np.int32), returnPoints=False)
            try:
                if len(hull) > 3:  
                    defects = cv2.convexityDefects(XY.astype(np.int32), hull)
                    if defects is not None and len(defects) > 5:
                        cv2.polylines(image, [XY.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            except cv2.error:
                continue
    return image

def detect_symmetry(image, paths_XYs):
    for XYs in paths_XYs:
        for XY in XYs:
            moments = cv2.moments(XY.astype(np.int32))
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                center = (cx, cy)
                for point in XY:
                    px, py = point
                    mirrored_point = (2 * cx - px, py)
                    cv2.circle(image, (int(mirrored_point[0]), int(mirrored_point[1])), 2, (255, 0, 0), -1)
    return image

def detect_occlusions(image, paths_XYs):
    contours = [XY.astype(np.int32) for XYs in paths_XYs for XY in XYs]
    for i, cnt1 in enumerate(contours):
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        for j, cnt2 in enumerate(contours):
            if i == j:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            if x1 < x2 < x1 + w1 and y1 < y2 < y1 + h1:
                cv2.drawContours(image, [cnt2], 0, (255, 0, 255), 2)
            elif (x1 < x2 < x1 + w1 or x2 < x1 < x2 + w2) and (y1 < y2 < y1 + h1 or y2 < y1 < y2 + h2):
                cv2.drawContours(image, [cnt2], 0, (0, 255, 255), 2)
            elif not (x1 < x2 < x1 + w1 and y1 < y2 < y1 + h1) and not (x2 < x1 < x2 + w2 and y2 < y1 < y2 + h2):
                cv2.drawContours(image, [cnt2], 0, (255, 0, 0), 2)
    return image

def process_image(image, paths_XYs):
    image_lines = detect_lines(image.copy(), paths_XYs)
    image_circles = detect_circles(image.copy(), paths_XYs)
    image_rectangles = detect_rectangles(image.copy(), paths_XYs)
    image_polygons = detect_polygons(image.copy(), paths_XYs)
    image_stars = detect_star_shapes(image.copy(), paths_XYs)
    image_symmetry = detect_symmetry(image.copy(), paths_XYs)
    image_occlusions = detect_occlusions(image.copy(), paths_XYs)

    plt.figure(figsize=(15, 10))

    plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(232), plt.imshow(cv2.cvtColor(image_lines, cv2.COLOR_BGR2RGB)), plt.title('Lines')
    plt.subplot(233), plt.imshow(cv2.cvtColor(image_circles, cv2.COLOR_BGR2RGB)), plt.title('Circles')
    plt.subplot(234), plt.imshow(cv2.cvtColor(image_rectangles, cv2.COLOR_BGR2RGB)), plt.title('Rectangles')
    plt.subplot(235), plt.imshow(cv2.cvtColor(image_polygons, cv2.COLOR_BGR2RGB)), plt.title('Polygons')
    plt.subplot(236), plt.imshow(cv2.cvtColor(image_stars, cv2.COLOR_BGR2RGB)), plt.title('Stars')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image_symmetry, cv2.COLOR_BGR2RGB))
    plt.title('Symmetry Detection')
    plt.show()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image_occlusions, cv2.COLOR_BGR2RGB))
    plt.title('Occlusion Detection')
    plt.show()

def polylines_to_svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
    dwg.add(group)
    dwg.save()

    
    png_path = svg_path.replace('.svg', '.png')
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=W, output_height=H, background_color='white')


csv_path = 'adobe genai\\test\\frag0.csv' 
paths_XYs = read_csv(csv_path)
plot(paths_XYs)

image_size = (500, 500, 3)
image = np.ones(image_size, dtype=np.uint8) * 255
svg_path = csv_path.split('.')[0]+"_svg.svg"
process_image(image, paths_XYs)
polylines_to_svg(paths_XYs, svg_path)