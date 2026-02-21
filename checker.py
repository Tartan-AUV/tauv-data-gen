import cv2
import sys

def visualize_yolo_pose(image_path, label_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    with open(label_path, 'r') as f:
        for line in f:
            data = list(map(float, line.split()))
            # data[0] is class, data[1:5] is bbox
            # data[5:] is keypoints: x1, y1, v1, x2, y2, v2...
            
            # Draw Bounding Box
            x_c, y_c, bw, bh = data[1], data[2], data[3], data[4]
            cv2.rectangle(img, (int((x_c-bw/2)*w), int((y_c-bh/2)*h)), 
                          (int((x_c+bw/2)*w), int((y_c+bh/2)*h)), (255, 0, 0), 2)

            # Draw Keypoints
            kpts = data[5:]
            for i in range(0, len(kpts), 3):
                kx, ky, kv = kpts[i], kpts[i+1], kpts[i+2]
                if kv > 0: # Only draw if visible/occluded (v=1 or 2)
                    color = (0, 255, 0) if kv == 2 else (0, 165, 255) # Green vs Orange
                    cv2.circle(img, (int(kx*w), int(ky*h)), 3, color, -1)
                    cv2.putText(img, str(i//3), (int(kx*w), int(ky*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Annotation Check", img)
    cv2.waitKey(0)

def main():
    visualize_yolo_pose("./output/rgb_" + sys.argv[1] + ".png", "./output/label_" + sys.argv[1] + ".txt")

if __name__ == "__main__":
    main()